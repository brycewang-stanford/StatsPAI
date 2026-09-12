"""Production functions vs Stata and R ``prodest`` (Rovigatti & Mollisi).

References: ``_fixtures/prodest_Stata.json`` (``_generate_prodest_stata.do``,
Stata 18 MP, SSC ``prodest``) and ``_fixtures/prodest_R.json``
(``_generate_prodest_R.R``, R ``prodest`` 1.0.2), both run on
``_fixtures/prodest_panel.csv`` (``_generate_prodest_data.py``: 250 firms,
unbalanced, 31 within-firm gaps, so every lag is a calendar lag).

What is, and is not, a parity claim
-----------------------------------
* OP / LP stage 1 (the free-input coefficient) is one OLS: 1e-12 against both
  references.
* OP / LP stage 2 (the state coefficient) is the minimiser of a sum of squares.
  Both references stop an optimiser early: Stata Nelder-Mead at tolerance 1e-5
  from OLS plus N(0, 0.01) noise, R BFGS from the first-stage coefficient plus
  the same noise. StatsPAI solves the first-order condition, has a sum of
  squares no larger than at either reference estimate, and lies within 1e-5
  (relative) of R's estimate and 1e-2 of Stata's. That is a
  reference-optimiser gap (T4), not parity.
* ACF: StatsPAI returns an exact root of the just-identified moment
  conditions, the one nearest the stage-1 coefficients. R stops within 1e-5 of
  that root, with criterion 1e-15. Stata stops at a point that is not a root
  (criterion ~1e-6). This data has three roots, and all are reported.
* WRDG, ``convention="prodest"``: Stata's stacked 2SLS, coefficients and
  unadjusted variance, to 1e-8 (observed 4e-9 on capital; within equation 1
  its column duplicates the h-polynomial's linear term, so Stata's solve
  loses a few digits). R's ``prodestWRDG`` drops the equation-2 intercept
  and leaves the constant out of the instrument set; rebuilt from StatsPAI's
  own design at 1e-12. Both impose a unit-slope Markov process. The default
  ``convention="statspai"`` estimates the slope by GMM; it has no reference
  implementation and is checked against the panel's known truth.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.structural.production import op_lp_acf as _oplp
from statspai.structural.production import wooldridge as _wrdg

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "prodest_R.json").read_text(encoding="utf-8"))
ST = json.loads((_FIX / "prodest_Stata.json").read_text(encoding="utf-8"))
KW = dict(output="y", free="l", state="k", panel_id="id", time="year")
PROXY = {"op": "lninv", "lp": "m"}


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(_FIX / "prodest_panel.csv")


def _fit_oplp(panel, method, degree):
    if method == "op":
        return sp.olley_pakes(
            panel, proxy="lninv", polynomial_degree=degree, drop_zero_proxy=False, **KW
        )
    return sp.levinsohn_petrin(panel, proxy="m", polynomial_degree=degree, **KW)


def _oplp_stage2(panel, method, degree):
    df = _oplp._prepare_panel(panel, "y", ["l"], ["k"], PROXY[method], "id", "year")
    return _oplp._OPLPStage2(df, "y", ["l"], ["k"], PROXY[method], degree, 3)


def _sse(stage, theta):
    return float(np.sum(stage.xi(np.array([theta]))[0] ** 2))


# --------------------------------------------------------------------------
# Olley-Pakes / Levinsohn-Petrin
# --------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("method", ["op", "lp"])
def test_oplp_free_input_coefficient_matches_both(panel, method, degree):
    r = _fit_oplp(panel, method, degree)
    np.testing.assert_allclose(
        r.coef["l"], ST[f"{method}_poly{degree}"]["b"]["l"], rtol=1e-12
    )
    assert (
        r.data_info["n_obs"] < ST[f"{method}_poly{degree}"]["N"]
    )  # stage 2 drops lag-less rows
    if degree == 2:  # R prodest hard-codes a degree-2 polynomial
        np.testing.assert_allclose(r.coef["l"], R[method]["pars"][0], rtol=1e-12)
        np.testing.assert_allclose(r.coef["l"], R[method]["fs_betas"][1], rtol=1e-12)


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("method", ["op", "lp"])
def test_oplp_state_coefficient_minimises_where_references_stop(panel, method, degree):
    r = _fit_oplp(panel, method, degree)
    theta = r.coef["k"]
    stage = _oplp_stage2(panel, method, degree)
    assert r.diagnostics["stage2_converged"]
    xi, _, _, J = stage._residual(np.array([theta]))
    cosine = abs(J[:, 0] @ xi) / (np.linalg.norm(J[:, 0]) * np.linalg.norm(xi))
    assert cosine < 1e-8

    stata_k = ST[f"{method}_poly{degree}"]["b"]["k"]
    assert _sse(stage, theta) <= _sse(stage, stata_k)
    assert abs(theta - stata_k) / abs(theta) < 1e-2  # observed 8e-4 .. 4.3e-3
    if degree == 2:
        r_k = R[method]["pars"][1]
        assert _sse(stage, theta) <= _sse(stage, r_k)
        # R's criterion at its own estimate, as recorded by R, is ours there.
        np.testing.assert_allclose(
            _sse(stage, r_k), R[method]["crit_at_pars"], rtol=1e-10
        )
        assert abs(theta - r_k) / abs(theta) < 1e-5  # observed 1.3e-6 / 5.9e-6


# --------------------------------------------------------------------------
# Ackerberg-Caves-Frazer
# --------------------------------------------------------------------------


def _acf(panel, degree):
    with pytest.warns(RuntimeWarning, match="distinct roots"):
        r = sp.ackerberg_caves_frazer(panel, proxy="m", polynomial_degree=degree, **KW)
    df = _oplp._prepare_panel(panel, "y", ["l"], ["k"], "m", "id", "year")
    stage = _oplp._ACFStage2(df, "y", ["l"], ["k"], "m", degree, 3, "cobb-douglas")
    return r, stage


@pytest.mark.parametrize("degree", [2, 3])
def test_acf_returns_the_root_nearest_the_first_stage(panel, degree):
    r, stage = _acf(panel, degree)
    beta = np.array([r.coef["l"], r.coef["k"]])
    assert stage.crit(beta) < 1e-25
    roots = np.array(r.diagnostics["acf_roots"])
    assert len(roots) >= 2
    assert all(stage.crit(root) < 1e-25 for root in roots)
    nearest = roots[np.argmin(np.linalg.norm(roots - stage.theta0, axis=1))]
    np.testing.assert_allclose(beta, nearest, rtol=0, atol=1e-10)

    stata = np.array(
        [ST[f"acf_poly{degree}"]["b"]["l"], ST[f"acf_poly{degree}"]["b"]["k"]]
    )
    assert stage.crit(stata) > 1e-8  # Stata's Nelder-Mead stops short of a root


def test_acf_r_stops_next_to_the_same_root(panel):
    r, stage = _acf(panel, 2)
    beta = np.array([r.coef["l"], r.coef["k"]])
    ref = np.array(R["acf"]["pars"])
    np.testing.assert_allclose(stage.crit(ref), R["acf"]["opt_value"], rtol=1e-8)
    assert stage.crit(ref) < 1e-12
    np.testing.assert_allclose(beta, ref, rtol=1e-5)  # observed 5e-7 / 2.4e-6


# --------------------------------------------------------------------------
# Wooldridge
# --------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [2, 3])
def test_wrdg_prodest_convention_matches_stata(panel, degree):
    r = sp.wooldridge_prod(
        panel,
        proxy="m",
        polynomial_degree=degree,
        vce="unadjusted",
        convention="prodest",
        **KW,
    )
    ref = ST[f"wrdg_poly{degree}"]
    for c in ("l", "k"):
        np.testing.assert_allclose(r.coef[c], ref["b"][c], rtol=1e-8)
        np.testing.assert_allclose(r.std_errors[c] ** 2, ref["V_diag"][c], rtol=1e-8)
    assert len(r.sample) == ref["N"]


def test_wrdg_r_variant_rebuilt_from_the_same_design(panel):
    """prodestWRDG: no equation-2 intercept, constant not an instrument, and
    the SE of lm() on the fitted regressors."""
    df = _oplp._prepare_panel(panel, "y", ["l"], ["k"], "m", "id", "year")
    design = _wrdg._wrdg_design(df, "y", ["l"], ["k"], "m", 2)
    X = np.delete(design["X"], [0, 1], axis=1)
    Z = np.delete(design["Z"], [0, 1], axis=1)
    Y = design["y"]
    Xhat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    D = np.column_stack([np.ones(len(Y)), Xhat])
    b = np.linalg.lstsq(D, Y, rcond=None)[0]
    e = Y - D @ b
    se = np.sqrt(np.diag(e @ e / (len(Y) - D.shape[1]) * np.linalg.inv(D.T @ D)))
    np.testing.assert_allclose(b[1:3], R["wrdg"]["pars"], rtol=1e-12)
    np.testing.assert_allclose(se[1:3], R["wrdg"]["se"], rtol=1e-10)

    # ...and it is not StatsPAI's prodest-convention estimate, which keeps e0.
    ours = sp.wooldridge_prod(
        panel, proxy="m", polynomial_degree=2, convention="prodest", **KW
    )
    assert abs(ours.coef["k"] - R["wrdg"]["pars"][1]) > 0.1


def test_wrdg_default_gmm_recovers_what_the_unit_slope_misses(panel):
    """The panel's productivity is AR(1) with rho = 0.7 and beta = (0.6, 0.3).
    Both prodest implementations impose rho = 1 and put capital near -0.6 to
    -0.8; the default GMM estimates g. Known truth, not a parity claim."""
    gmm = sp.wooldridge_prod(panel, proxy="m", **KW)
    assert gmm.coef["l"] == pytest.approx(0.60, abs=0.05)
    assert gmm.coef["k"] == pytest.approx(0.30, abs=0.10)
    assert gmm.productivity_process["rho"] == pytest.approx(0.70, abs=0.05)
    assert ST["wrdg_poly3"]["b"]["k"] < 0 and R["wrdg"]["pars"][1] < 0
