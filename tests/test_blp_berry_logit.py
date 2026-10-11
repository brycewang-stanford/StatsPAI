"""Berry (1994) logit and nested logit through ``sp.blp``: identities,
known truth and the input contract. The pyblp comparison is in
``reference_parity/test_berry_logit_pyblp_parity.py``."""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_GEN = Path(__file__).parent / "reference_parity" / "_generate_berry_logit_data.py"
_spec = importlib.util.spec_from_file_location("_berry_dgp", _GEN)
_dgp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dgp)
KW = dict(shares="share", prices="price", x_linear=["x1", "x2"], x_random=[])


@pytest.fixture(scope="module")
def data():
    return _dgp.simulate(120, seed=1994)


def _with_y(d):
    d = d.copy()
    d["y"] = np.log(d["share"]) - np.log(
        1.0 - d.groupby("market_id")["share"].transform("sum")
    )
    return d


# ---- instruments -----------------------------------------------------------


def test_blp_instruments_by_hand():
    d = pd.DataFrame(
        {
            "market_id": [1, 1, 1, 1, 2, 2],
            "firm": ["a", "a", "b", "c", "a", "b"],
            "nest": ["s", "t", "s", "s", "s", "t"],
            "x": [1.0, 2.0, 4.0, 8.0, 3.0, 5.0],
        },
        index=list("pqrstu"),
    )
    z = sp.blp_instruments(d, ["x"], firm_id="firm", nest="nest")
    assert list(z.index) == list("pqrstu")
    assert z["own_x"].tolist() == [2.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert z["rival_x"].tolist() == [12.0, 12.0, 11.0, 7.0, 5.0, 3.0]
    assert z["nest_x"].tolist() == [12.0, 0.0, 9.0, 5.0, 0.0, 0.0]
    assert z["own_count"].tolist() == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    assert z["rival_count"].tolist() == [2.0, 2.0, 3.0, 3.0, 1.0, 1.0]
    assert z["nest_count"].tolist() == [2.0, 0.0, 2.0, 2.0, 0.0, 0.0]
    # no firm: every other product of the market is a rival
    assert sp.blp_instruments(d, ["x"])["rival_x"].tolist() == [
        14.0,
        13.0,
        11.0,
        7.0,
        5.0,
        3.0,
    ]
    assert list(sp.blp_instruments(d, ["x"], count=False).columns) == ["rival_x"]


def test_blp_instruments_rejects_bad_input():
    d = pd.DataFrame({"market_id": [1, 1], "x": [1.0, np.nan]})
    with pytest.raises(ValueError, match="missing values"):
        sp.blp_instruments(d, ["x"])
    with pytest.raises(ValueError, match="not found"):
        sp.blp_instruments(d, ["x"], firm_id="firm")
    with pytest.raises(ValueError, match="at least one"):
        sp.blp_instruments(d, [], count=False)


# ---- identities ------------------------------------------------------------


def test_one_step_logit_is_two_stage_least_squares(data):
    """W = (Z'Z)^{-1} on a linear model is 2SLS, with the HC0 sandwich."""
    res = sp.blp(data, firm_id="firm", gmm_steps=1, **KW)
    z = sp.blp_instruments(data, ["x1", "x2"], firm_id="firm")
    assert res.instruments == list(z.columns) and res.model == "logit"
    d = pd.concat([_with_y(data), z], axis=1)
    iv = sp.iv(f"y ~ (price ~ {' + '.join(z.columns)}) + x1 + x2", data=d, robust="hc0")
    order = ["Intercept", "x1", "x2", "price"]
    np.testing.assert_allclose(res.linear_params.values, iv.params[order], rtol=1e-9)
    np.testing.assert_allclose(res.se_linear.values, iv.std_errors[order], rtol=1e-9)
    np.testing.assert_allclose(res.mean_utility.values, d["y"].values, atol=1e-12)


def test_one_step_nested_logit_is_two_stage_least_squares(data):
    res = sp.blp(data, firm_id="firm", nest="nest", gmm_steps=1, **KW)
    z = sp.blp_instruments(data, ["x1", "x2"], firm_id="firm", nest="nest")
    d = pd.concat([_with_y(data), z], axis=1)
    d["ln_within"] = np.log(
        d["share"] / d.groupby(["market_id", "nest"])["share"].transform("sum")
    )
    iv = sp.iv(
        f"y ~ (price + ln_within ~ {' + '.join(z.columns)}) + x1 + x2",
        data=d,
        robust="hc0",
    )
    order = ["Intercept", "x1", "x2", "price"]
    np.testing.assert_allclose(res.linear_params.values, iv.params[order], rtol=1e-9)
    assert res.rho == pytest.approx(iv.params["ln_within"], rel=1e-9)
    assert res.se_nonlinear["rho"] == pytest.approx(
        iv.std_errors["ln_within"], rel=1e-9
    )
    assert list(res.nonlinear_params.index) == ["rho"]


def _nested_shares(delta, nest, rho):
    ev = np.exp(delta / (1.0 - rho))
    D = {g: ev[nest == g].sum() for g in np.unique(nest)}
    denom = 1.0 + sum(v ** (1.0 - rho) for v in D.values())
    return ev / np.array([D[g] for g in nest]) ** rho / denom


@pytest.mark.parametrize("nested", [False, True])
def test_elasticities_are_derivatives_of_the_share_function(data, nested):
    """Central differences of the (nested) logit shares in each price."""
    res = sp.blp(
        data, firm_id="firm", nest="nest" if nested else None, gmm_steps=1, **KW
    )
    m = data[data["market_id"] == 3]
    idx = m.index.to_numpy()
    rho = res.rho if nested else 0.0
    alpha = res.linear_params["price"]
    nest = m["nest"].to_numpy()
    delta, p = res.mean_utility.values[idx], m["price"].to_numpy()
    s = _nested_shares(delta, nest, rho)
    np.testing.assert_allclose(s, m["share"].to_numpy(), rtol=1e-10)
    E = np.empty((len(idx), len(idx)))
    h = 1e-6
    for k in range(len(idx)):
        step = np.zeros(len(idx))
        step[k] = alpha * h
        ds = _nested_shares(delta + step, nest, rho) - _nested_shares(
            delta - step, nest, rho
        )
        E[:, k] = ds / (2 * h) * p[k] / s
    np.testing.assert_allclose(res.elasticity_matrix(3).values, E, rtol=1e-6)


def test_logit_own_elasticity_closed_form(data):
    res = sp.blp(data, firm_id="firm", **KW)
    a = res.linear_params["price"]
    np.testing.assert_allclose(
        res.own_elasticities.values,
        a * data["price"] * (1.0 - data["share"]),
        rtol=1e-10,
    )


# ---- known truth -----------------------------------------------------------


def test_nested_logit_recovers_the_data_generating_parameters():
    """Exact nested-logit shares, 600 markets: every parameter within 3
    robust standard errors of the truth, and the plain logit (which leaves
    rho * ln s_{j|g} in the error) is far from the price coefficient."""
    d = _dgp.simulate(600, seed=7)
    z = sp.blp_instruments(d, ["x1", "x2"], firm_id="firm", nest="nest")
    d = pd.concat([d, z], axis=1)
    iv = list(z.columns) + ["w"]
    res = sp.blp(d, instruments=iv, nest="nest", **KW)
    truth = _dgp.TRUE
    for name, key in (
        ("const", "const"),
        ("x1", "x1"),
        ("x2", "x2"),
        ("price", "price"),
    ):
        assert abs(res.linear_params[name] - truth[key]) < 3 * res.se_linear[name]
    assert abs(res.rho - truth["rho"]) < 3 * res.se_nonlinear["rho"]
    assert res.se_nonlinear["rho"] < 0.02
    assert res.hansen_p > 0.01
    logit = sp.blp(d, instruments=[c for c in iv if not c.startswith("nest_")], **KW)
    assert logit.linear_params["price"] < 1.5 * truth["price"]


# ---- contract --------------------------------------------------------------


def test_default_instruments_with_random_coefficients_run(data):
    """The default instruments used to repeat x_linear, and the weighting
    matrix was the inverse of a singular Z'Z."""
    res = sp.blp(
        data[data["market_id"] < 25],
        shares="share",
        prices="price",
        x_linear=["x1", "x2"],
        x_random=["x1"],
        firm_id="firm",
        n_draws=10,
        maxiter=8,
        tol_inner=1e-8,
        seed=0,
    )
    assert res.model == "random_coefficients"
    assert np.all(np.isfinite(res.se_linear.values))
    assert np.all(np.abs(res.linear_params.values) < 50)
    assert res.omitted_instruments == []


def test_redundant_default_instrument_is_dropped_silently():
    """Same number of products in every market: the rival count is the
    constant."""
    rng = np.random.default_rng(0)
    rows = []
    for t in range(60):
        x = rng.normal(size=4)
        xi = rng.normal(0, 0.3, 4)
        p = 1 + 0.5 * rng.uniform(size=4) + 0.4 * xi
        ev = np.exp(0.5 + x - p + xi)
        rows.append(
            pd.DataFrame(
                {
                    "market_id": t,
                    "product_id": range(4),
                    "share": ev / (1 + ev.sum()),
                    "price": p,
                    "x1": x,
                }
            )
        )
    d = pd.concat(rows, ignore_index=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sp.blp(d, shares="share", prices="price", x_linear=["x1"], x_random=[])
    assert res.instruments == ["rival_x1"]
    assert [o["variable"] for o in res.omitted_instruments] == ["rival_count"]
    assert res.hansen_j is None  # just identified


def test_redundant_supplied_instrument_warns(data):
    d = data.assign(w2=lambda f: 2.0 * f["w"])
    with pytest.warns(UserWarning, match="instrument w2 omitted"):
        res = sp.blp(d, instruments=["w", "w2"], **KW)
    assert res.instruments == ["w"]


def test_refusals(data):
    with pytest.raises(sp.MethodIncompatibility, match="x_random=\\[\\]"):
        sp.blp(
            data,
            shares="share",
            prices="price",
            x_linear=["x1", "x2"],
            x_random=["x1"],
            nest="nest",
        )
    with pytest.raises(sp.DataInsufficient, match="within-nest share"):
        sp.blp(data, instruments=["w"], nest="nest", **KW)
    with pytest.raises(ValueError, match="gmm_steps"):
        sp.blp(data, gmm_steps=3, **KW)
    with pytest.raises(sp.MethodIncompatibility, match="collinear"):
        sp.blp(
            data.assign(x3=lambda f: f["x1"] + f["x2"]),
            shares="share",
            prices="price",
            x_linear=["x1", "x2", "x3"],
            x_random=[],
        )
    full = data.copy()
    full["share"] = full["share"] / full.groupby("market_id")["share"].transform("sum")
    full["share"] = full["share"].clip(upper=0.999)
    with pytest.raises(ValueError, match="outside good"):
        sp.blp(full[full["market_id"] < 5], **KW)


def test_nesting_parameter_outside_unit_interval_warns():
    """Plain logit shares (rho = 0): the estimate is zero up to sampling
    error, and when it falls below zero the result says so."""
    d = _dgp.simulate(60, seed=0, rho=0.0)
    with pytest.warns(UserWarning, match=r"outside \[0, 1\)"):
        res = sp.blp(d, firm_id="firm", nest="nest", **KW)
    assert res.rho < 0.0
    assert abs(res.rho) < 2 * res.se_nonlinear["rho"]


def test_nest_of_single_products_is_refused(data):
    with pytest.raises(sp.MethodIncompatibility, match="alone in its nest"):
        sp.blp(data.assign(solo=data["product_id"]), nest="solo", **KW)


def test_summary_and_conversion_name_the_model(data):
    res = sp.blp(data, firm_id="firm", nest="nest", **KW)
    text = res.summary()
    assert "Nested Logit (Berry 1994)" in text and "Hansen J" in text
    er = res.to_econometric_results()
    assert "rho" in er.params.index
    assert er.model_info["instruments"] == res.instruments
    assert er.diagnostics["Hansen J df"] == res.hansen_df
