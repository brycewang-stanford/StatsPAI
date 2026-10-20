"""``sp.scpi`` conventions pinned against R ``scpi`` 4.0.1 used as a black box.

``sp.scdata`` / ``sp.scest`` / ``sp.scpi`` were rewritten clean-room from the
scpi papers, the R help pages and black-box runs of R ``scpi`` (GPL-3); the
specification is ``docs/dev/cleanroom/scpi.md``. This file checks the
conventions that the papers leave open and that the Germany fixture
(``test_did_synth_scpi_parity.py``) does not exercise, on two committed
synthetic panels (``_fixtures/_generate_scpi_cleanroom_{data.py,R.R}``):

* the rho rules (type-1, a user value, a value above every weight);
* the radii of the simulation constraint set (lasso binding by the rho rule
  but slack in fact, lasso slack, user ridge radius, a ridge radius
  enlarged to ``||w||^2 + rho``, and a binding L1-L2 ball that stays at Q2
  even with rho = 0.3, where the ridge rule would enlarge it);
* HC2 / HC4 leverage and degrees of freedom, ``u.missp = FALSE``,
  ``u.order = 0``, non-default levels;
* the constant out-of-sample design (``e.order = 0``, and the fallback for
  short panels), where the quantile bounds are the sample quantiles;
* the data-driven ridge radius on a screened design (``T0 - J <= 10``).

Tolerances, as in ``test_did_synth_scpi_parity.py``: deterministic
ingredients 1e-9 relative; out-of-sample bounds 1e-9 against R re-run with
``rrq(method = "br")`` (exact LP); per-draw in-sample values against ECOS at
1e-12 (median 1e-6, max 2e-4); bounds as an R user gets them (ECOS 1e-8,
Frisch-Newton rrq) within 2e-3. The quantile and simultaneous-bound
constructions are also checked as identities on R's own per-draw values.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.synth.scpi import _constraint_spec, scpi_inference

FIX = Path(__file__).parent / "_fixtures"
R = json.loads((FIX / "scpi_cleanroom_R.json").read_text(encoding="utf-8"))
CASES = sorted(k for k in R if k != "versions")
PANELS = {
    "long": ("scpi_cleanroom_long.csv", 18),
    "short": ("scpi_cleanroom_short.csv", 13),
    "tiny": ("scpi_cleanroom_tiny.csv", 4),
}
OPT_NAMES = {
    "e.order": "e_order",
    "u.order": "u_order",
    "u.sigma": "u_sigma",
    "u.missp": "u_missp",
    "u.alpha": "u_alpha",
    "e.alpha": "e_alpha",
    "rho": "rho",
}


def _arr(x):
    return np.asarray(x, dtype=float)


def _close(ours, ref, rtol=1e-9, atol=0.0):
    np.testing.assert_allclose(_arr(ours), _arr(ref), rtol=rtol, atol=atol)


_DATA: dict = {}
_INF: dict = {}


def _data(panel):
    if panel not in _DATA:
        name, t0 = PANELS[panel]
        df = pd.read_csv(FIX / name)
        _DATA[panel] = sp.scdata(df, "y", "unit", "time", "tr", t0)
    return _DATA[panel]


def _spec(case):
    r = R[case]
    sc = _data(r["panel"])
    wc = r["w_constr"]
    return _constraint_spec(sc["A"], sc["B"], wc["name"], wc.get("Q"), wc.get("Q2"))


def _opts(case):
    return {OPT_NAMES[k]: v for k, v in (R[case]["opt"] or {}).items()}


def _inference(case):
    """StatsPAI's inference on R's weights and R's draws."""
    if case not in _INF:
        r = R[case]
        sc = _data(r["panel"])
        z = _arr(r["zraw"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _INF[case] = scpi_inference(
                sc["A"],
                sc["B"],
                sc["P"],
                _arr(r["w"]),
                _spec(case),
                sims=z.shape[1],
                draws=z,
                **_opts(case),
            )
    return _INF[case]


def _e_part(bounds, key):
    return _arr(bounds[key]) - _arr(bounds["insample"])


@pytest.mark.parametrize("case", CASES)
def test_constraint_radius_matches(case):
    r = R[case]
    spec = _spec(case)
    if r["Q"] is not None:
        _close(spec["Q"], r["Q"], rtol=1e-10)
    if r["Q2"] is not None:
        _close(spec["Q2"], r["Q2"], rtol=1e-10)
    if r["lambda"] is None:
        assert spec["lambda"] is None
    else:
        _close(spec["lambda"], r["lambda"], rtol=1e-10)


@pytest.mark.parametrize("case", CASES)
def test_weights_optimal_within_the_solver_gap(case):
    r = R[case]
    sc = _data(r["panel"])
    A, B = sc["A"], sc["B"]
    assert sc["donor_names"] == r["donors"]
    wc = r["w_constr"]
    est = sp.scest(
        pd.read_csv(FIX / PANELS[r["panel"]][0]),
        "y",
        "unit",
        "time",
        "tr",
        PANELS[r["panel"]][1],
        w_constr=wc["name"],
        Q=wc.get("Q"),
        Q2=wc.get("Q2"),
    )
    w_py, w_R = est["weights"], _arr(r["scest_w"])
    # R's conic solver may overshoot a norm bound by its feasibility
    # tolerance; pull its point back onto the constraint set first
    if wc["name"] == "lasso":
        w_R = w_R * min(1.0, wc["Q"] / np.abs(w_R).sum())
    if wc["name"] == "ridge":
        w_R = w_R * min(1.0, est["w_constr_spec"]["Q"] / np.linalg.norm(w_R))
    if wc["name"] == "L1-L2":
        # move toward the uniform weights (inside the ball) until the norm
        # meets Q2; the simplex constraints are kept
        Q2 = est["w_constr_spec"]["Q2"]
        u = np.full(len(w_R), 1.0 / len(w_R))
        if np.linalg.norm(w_R) > Q2:
            d = w_R - u
            t = (-(u @ d) + np.sqrt((u @ d) ** 2 - (d @ d) * (u @ u - Q2**2))) / (d @ d)
            w_R = u + t * d

    def f(w):
        return float(np.sum((A - B @ w) ** 2))

    gap = f(w_R) - f(w_py)
    assert gap >= -1e-14 * f(w_py)  # the exact optimum is never worse
    assert gap <= 1e-7 * f(w_py)  # the conic solver's relative gap
    lam_min = np.linalg.eigvalsh(B.T @ B).min()
    assert np.linalg.norm(w_R - w_py) <= np.sqrt(max(gap, 0.0) / lam_min) + 1e-9


@pytest.mark.parametrize("case", CASES)
def test_deterministic_ingredients(case):
    r = R[case]
    out = _inference(case)
    _close(out["rho"], r["rho"])
    if r["Q_star"] is not None:
        _close(out["Q_star"], r["Q_star"])
    _close(out["u_mean"], r["u_mean"], rtol=1e-9, atol=1e-13)
    _close(out["u_var"], r["u_var"], rtol=1e-9)
    S = _arr(r["Sigma"])
    _close(out["Sigma"], S, rtol=1e-9, atol=1e-12 * np.abs(S).max())
    assert (out["u_T"], out["u_params"], out["u_order"]) == (
        r["u_T"],
        r["u_params"],
        r["u_order"],
    )
    assert (out["e_T"], out["e_params"], out["e_order"]) == (
        r["e_T"],
        r["e_params"],
        r["e_order"],
    )
    _close(out["e_mean"], r["e_mean"], rtol=1e-9, atol=1e-12)
    _close(out["e_var"], r["e_var_br"], rtol=1e-9, atol=1e-14)


@pytest.mark.parametrize("case", CASES)
def test_out_of_sample_bounds_vs_exact_lp(case):
    r = R[case]
    out = _inference(case)
    for ours, theirs in (
        ("subgaussian", "subgaussian"),
        ("ls", "ls"),
        ("qreg", "qreg"),
    ):
        _close(
            out["e_bounds"][ours], _e_part(r["bounds_br"], theirs), rtol=0, atol=1e-9
        )


def test_constant_design_quantile_bounds_are_sample_quantiles():
    """Known truth: with e.order = 0 the quantile bounds are the type-7
    sample quantiles of the pre-treatment residuals, in every period."""
    r = R["simplex_e_order0"]
    sc = _data(r["panel"])
    res = sc["A"] - sc["B"] @ _arr(r["w"])
    e_q = _e_part(r["bounds"], "qreg")
    want = np.quantile(res, [0.025, 0.975])
    np.testing.assert_allclose(e_q, np.tile(want, (len(e_q), 1)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        _inference("simplex_e_order0")["e_bounds"]["qreg"], e_q, rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("case", CASES)
def test_insample_per_draw_vs_tight_ecos(case):
    r = R[case]
    out = _inference(case)
    tight = _arr(r["vsig_tight"])
    ours = out["vsig"]
    assert out["n_slsqp_fallback"] == 0
    assert np.isfinite(ours).all() and np.isfinite(tight).all()
    d = np.abs(ours - tight)
    assert np.median(d) <= 1e-6
    assert d.max() <= 2e-4


@pytest.mark.parametrize("case", CASES)
def test_quantile_and_joint_constructions_on_R_draws(case):
    """Identities on R's own per-draw values and out-of-sample bounds:
    in-sample bounds are type-7 quantiles per period, and the simultaneous
    bounds are the quantiles of the per-draw extremes over periods plus
    sqrt(log(T1 + 1)) times the sub-Gaussian out-of-sample bounds."""
    r = R[case]
    opts = _opts(case)
    a1 = opts.get("u_alpha", 0.05)
    v = _arr(r["vsig"])
    T1 = v.shape[1] // 2
    lo, hi = v[:, :T1], v[:, T1:]
    ins = _arr(r["bounds"]["insample"])
    np.testing.assert_allclose(
        ins[:, 0], np.quantile(lo, a1 / 2, axis=0), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        ins[:, 1], np.quantile(hi, 1 - a1 / 2, axis=0), rtol=0, atol=1e-12
    )
    eg = _e_part(r["bounds"], "subgaussian")
    k = np.sqrt(np.log(T1 + 1))
    joint = _arr(r["bounds"]["joint"])
    np.testing.assert_allclose(
        joint[:, 0], np.quantile(lo.min(1), a1 / 2) + k * eg[:, 0], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        joint[:, 1],
        np.quantile(hi.max(1), 1 - a1 / 2) + k * eg[:, 1],
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("case", CASES)
def test_bounds_as_an_R_user_gets_them(case):
    r = R[case]
    out = _inference(case)
    for k in ("insample", "subgaussian", "ls", "qreg", "joint"):
        _close(out["bounds"][k], r["bounds"][k], rtol=0, atol=2e-3)
    assert np.all(_arr(r["failed_sims"]) == 0)
    assert np.all(out["failed_sims"] == 0)


def test_rho_above_every_weight_keeps_the_largest_donor():
    """R: with rho = 5 the residual designs use one donor (the largest
    weight) and a constant; the simplex lower bounds are the weights."""
    r = R["simplex_rho5"]
    assert r["u_params"] == 2 and r["e_params"] == 2
    with pytest.warns(RuntimeWarning, match="rho was too high"):
        sc = _data(r["panel"])
        out = scpi_inference(
            sc["A"],
            sc["B"],
            sc["P"],
            _arr(r["w"]),
            _spec("simplex_rho5"),
            sims=10,
            rho=5,
        )
    np.testing.assert_allclose(out["lb"], _arr(r["w"]), rtol=0, atol=0)


def test_lasso_radius_rule():
    """R's Q.star: ||w||_1 + rho when ||w||_1 > Q - rho (binding by the rule,
    even when slack in fact, Q = 1.3), else Q (Q = 1.5)."""
    for case, binding in (
        ("lasso_Q13", True),
        ("lasso_Q15", False),
        ("lasso_Q05", True),
    ):
        r = R[case]
        l1 = np.abs(_arr(r["w"])).sum()
        want = l1 + r["rho"] if binding else r["Q"]
        _close(r["Q_star"], want, rtol=1e-12)
        _close(_inference(case)["Q_star"], want, rtol=1e-12)


def test_ridge_radius_rule():
    """R's Q.star for the L2 ball: ||w||^2 + rho when ||w||^2 > Q - rho (the
    squared norm against Q; ridge_short: 0.909 > 0.953 - 0.2), else Q
    (ridge_Q03: 0.09 < 0.1). The simulated values use that radius
    (test_insample_per_draw_vs_tight_ecos)."""
    for case, binding in (("ridge_short", True), ("ridge_Q03", False)):
        r = R[case]
        sq = float(_arr(r["w"]) @ _arr(r["w"]))
        assert (sq > r["Q"] - r["rho"]) is binding
        want = sq + r["rho"] if binding else r["Q"]
        _close(r["Q_star"], want, rtol=1e-12)
        _close(_inference(case)["Q_star"], want, rtol=1e-12)


def test_l1l2_ball_rule():
    """The L1-L2 ball follows the ridge rule with min(rho, rho_max): enlarged
    to ||w||^2 + 0.2 on the T0 = 3 panel, kept at Q2 with rho = 0.3 (where
    min(0.3, 0.2) leaves it slack by the rule). R does not report Q2*; the
    per-draw parity of both cases (test_insample_per_draw_vs_tight_ecos)
    is the evidence."""
    r = R["L1L2_tiny"]
    sq = float(_arr(r["w"]) @ _arr(r["w"]))
    _close(_inference("L1L2_tiny")["Q2_star"], sq + 0.2, rtol=1e-12)
    r = R["L1L2_Q2_066_rho03"]
    sq = float(_arr(r["w"]) @ _arr(r["w"]))
    assert sq <= r["Q2"] - 0.2
    _close(_inference("L1L2_Q2_066_rho03")["Q2_star"], r["Q2"], rtol=1e-12)
