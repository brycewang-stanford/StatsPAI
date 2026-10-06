"""Markov-switching regression against Stata 18 ``mswitch`` and statsmodels.

Reference: ``_fixtures/mswitch_Stata_1.json`` to ``_4.json``, written by
``_fixtures/_generate_mswitch_Stata.do`` from the simulated
``_fixtures/mswitch.csv`` (``_generate_mswitch_data.py``).

The comparison is made in two steps. First our likelihood, filter,
smoother and covariance are evaluated *at Stata's estimates*, where the
only differences are rounding and Stata's numerical derivatives. Then the
optimum we find is compared with Stata's, where the difference is Stata's
stopping rule (``nrtolerance(1e-10)``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from statspai.timeseries import _mswitch_core as core
from statspai.timeseries.mswitch import mswitch

FIX = Path(__file__).parent / "_fixtures"
DATA = pd.read_csv(FIX / "mswitch.csv")
REF: Dict[str, Any] = {}
for _part in sorted(FIX.glob("mswitch_Stata_[0-9].json")):
    REF.update(json.loads(_part.read_text(encoding="utf-8")))

CASES: Dict[str, Dict[str, Any]] = {
    "mean": dict(y="y_mean"),
    "var": dict(y="y_var", switch_variance=True),
    "x": dict(y="y_x", x="x"),
    "z": dict(y="y_z", switch="z"),
    "k3": dict(y="y3", states=3),
    "common": dict(y="y_z", switch="z", constant="common"),
    "drlag": dict(y="y_ar", ar=1),
    "drlagsw": dict(y="y_ar", ar=2, switch_ar=True),
    "varrob": dict(y="y_var", switch_variance=True, vce="robust"),
    "ar1": dict(y="y_ar", model="ar", ar=1),
    "ar2": dict(y="y_ar", model="ar", ar=2),
    "ar4": dict(y="y_ar", model="ar", ar=4),
    "ar1x": dict(y="y_x", x="x", model="ar", ar=1),
    "ar1z": dict(y="y_z", switch="z", model="ar", ar=1),
    "arsw": dict(y="y_ar", model="ar", ar=1, switch_ar=True),
    "arvar": dict(y="y_var", model="ar", ar=1, switch_variance=True),
    "ar2var": dict(y="y_var", model="ar", ar=2, switch_variance=True),
    "k3ar1": dict(y="y3_ar", model="ar", ar=1, states=3),
}
# Stata stopped short of a maximum here (its gradient is far from zero and
# one transition probability is exactly 0); only the likelihood and the
# filter at its parameter vector are compared.
STATA_NOT_CONVERGED = {"k3ar1"}
# Stata's -predict- after -mswitch ar ..., switch()- does not reproduce the
# probabilities of the likelihood it maximised; statsmodels agrees with us
# (test_ar_switching_regressor_probabilities_follow_statsmodels).
STATA_PREDICT_DIFFERS = {"ar1z"}


def _stata_names(kw: Dict[str, Any]) -> List[str]:
    """Stata's e(b) column names in the order of our parameter vector."""
    dep, k, p = kw["y"], kw.get("states", 2), kw.get("ar", 0)
    stem = "ar" if kw.get("model", "dr") == "ar" else dep

    def lag(j: int) -> str:
        return ("L" if j == 1 else f"L{j}") + "." + stem

    states = [f"State{s + 1}" for s in range(k)]
    out = (
        [f"{s}:_cons" for s in states]
        if kw.get("constant", True) is True
        else [f"{dep}:_cons"]
    )
    if "x" in kw:
        out.append(f"{dep}:{kw['x']}")
    if "switch" in kw:
        out += [f"{s}:{kw['switch']}" for s in states]
    if kw.get("switch_ar"):
        out += [f"{s}:{lag(j)}" for s in states for j in range(1, p + 1)]
    else:
        out += [f"{dep}:{lag(j)}" for j in range(1, p + 1)]
    if kw.get("switch_variance"):
        out += [f"lnsigma{s + 1}:_cons" for s in range(k)]
    else:
        out.append("lnsigma:_cons")
    out += [f"p{i + 1}{j + 1}:_cons" for i in range(k) for j in range(k - 1)]
    return out


def _setup(key: str) -> Dict[str, Any]:
    kw, ref = CASES[key], REF[key]
    idx = [ref["names"].index(n) for n in _stata_names(kw)]
    assert sorted(idx) == list(range(len(ref["names"])))
    p = kw.get("ar", 0)
    spec = core.Spec(
        kw.get("states", 2),
        kw.get("model", "dr") if p else "dr",
        p,
        int("x" in kw),
        int("switch" in kw),
        "switch" if kw.get("constant", True) is True else "common",
        kw.get("switch_ar", False),
        kw.get("switch_variance", False),
    )
    none = np.empty((len(DATA), 0))
    dat = core.Data(
        DATA[kw["y"]].to_numpy(),
        DATA[[kw["x"]]].to_numpy() if "x" in kw else none,
        DATA[[kw["switch"]]].to_numpy() if "switch" in kw else none,
        p,
    )
    theta = np.array(ref["b"][0], dtype=float)[idx]
    vcov = np.array(ref["V"], dtype=float)[np.ix_(idx, idx)]
    return {"kw": kw, "ref": ref, "spec": spec, "dat": dat, "b": theta, "V": vcov}


@pytest.mark.parametrize("key", list(CASES))
def test_loglik_at_stata_estimates(key: str) -> None:
    s = _setup(key)
    ll = core.loglik(s["spec"], s["dat"], s["b"])
    # same filter, same ergodic initial distribution: rounding only
    assert ll == pytest.approx(s["ref"]["ll"], rel=1e-12)


@pytest.mark.parametrize("key", [k for k in CASES if k not in STATA_PREDICT_DIFFERS])
def test_probabilities_at_stata_estimates(key: str) -> None:
    s = _setup(key)
    ref = s["ref"]
    fs = core.filter_smooth(s["spec"], s["dat"], s["b"])
    # rounding of a 300-step recursion (up to 2e-13 with 32 regimes)
    np.testing.assert_allclose(fs["pred"], np.array(ref["predicted"]), atol=1e-11)
    np.testing.assert_allclose(fs["filt"], np.array(ref["filtered"]), atol=1e-11)
    yv = s["dat"].y[s["spec"].p :]
    yhat = (fs["pred_exp"] * (yv[:, None] - fs["resid_exp"])).sum(axis=1)
    np.testing.assert_allclose(yhat, np.array(ref["yhat_filter"][0]), atol=1e-11)
    if key not in STATA_NOT_CONVERGED:
        np.testing.assert_allclose(fs["smooth"], np.array(ref["smoothed"]), atol=1e-11)


def test_stata_default_yhat_applies_the_transition_once_more() -> None:
    s = _setup("mean")
    fs = core.filter_smooth(s["spec"], s["dat"], s["b"])
    mu = s["b"][:2]
    ours = fs["pred"] @ mu
    stata = np.array(s["ref"]["yhat_default"][0])
    np.testing.assert_allclose(ours[0], stata[0], atol=1e-12)
    assert np.abs(ours[1:] - stata[1:]).max() > 0.05
    np.testing.assert_allclose((fs["pred"] @ fs["P"] @ mu)[1:], stata[1:], atol=1e-12)


@pytest.mark.parametrize("key", [k for k in CASES if k not in STATA_NOT_CONVERGED])
def test_covariance_at_stata_estimates(key: str) -> None:
    s = _setup(key)
    vcov, notes = core.covariance(
        s["spec"], s["dat"], s["b"], s["kw"].get("vce", "oim")
    )
    assert notes == []
    scale = np.sqrt(np.outer(np.diag(s["V"]), np.diag(s["V"])))
    # Stata differentiates numerically (we use complex-step scores): the
    # two Hessians agree to about 1e-6 of the standard errors
    assert (np.abs(vcov - s["V"]) / scale).max() < 5e-6


@pytest.mark.parametrize(
    "key", ["mean", "var", "x", "z", "k3", "common", "drlagsw", "ar1", "ar2", "arsw"]
)
def test_optimum_matches_stata(key: str) -> None:
    s = _setup(key)
    ref = s["ref"]
    fit = mswitch(data=DATA, seed=1, **s["kw"])
    assert fit.converged
    # we stop at a Newton decrement of 1e-9 with exact scores; Stata at
    # nrtolerance(1e-10) with numerical ones, so ours is not lower
    assert fit.loglik >= ref["ll"] - 1e-9
    assert fit.loglik == pytest.approx(ref["ll"], abs=1e-8)
    target, _ = core.reorder(s["spec"], s["b"])
    np.testing.assert_allclose(target, s["b"])  # Stata's order is ours here
    se = np.sqrt(np.diag(fit.vcov.to_numpy()))
    # Stata's stopping rule leaves it within about 1e-5 standard errors
    assert (np.abs(fit.theta.to_numpy() - target) / se).max() < 5e-5
    tt = fit.transition_table
    np.testing.assert_allclose(tt["estimate"], ref["prob"][0], atol=2e-6)
    np.testing.assert_allclose(tt["se"], ref["prob_se"][0], rtol=2e-5)
    np.testing.assert_allclose(
        tt[["ci_lower", "ci_upper"]].to_numpy(), np.array(ref["prob_ci"]), atol=2e-6
    )
    dur = np.array(ref["duration"])
    np.testing.assert_allclose(fit.durations["estimate"], dur[0], rtol=2e-5)
    np.testing.assert_allclose(fit.durations["se"], dur[1], rtol=2e-5)
    got = fit.durations[["ci_lower", "ci_upper"]].to_numpy().T
    if fit.states == 2:
        np.testing.assert_allclose(got, dur[2:], rtol=2e-5)
    else:
        # with three states Stata reports estimate +/- z * se instead of
        # the transformed interval of p_ii; rebuild it from our numbers
        z = 1.959963984540054
        d = fit.durations
        np.testing.assert_allclose(d["estimate"] - z * d["se"], dur[2], rtol=2e-5)
        np.testing.assert_allclose(d["estimate"] + z * d["se"], dur[3], rtol=2e-5)
        assert (got[0] > dur[2]).all()


def test_three_state_ar_beats_stata_end_point() -> None:
    s = _setup("k3ar1")
    grad = core.gradient(s["spec"], s["dat"], s["b"])
    assert np.abs(grad).max() > 1.0  # Stata's reported point is not stationary
    fit = mswitch(data=DATA, seed=1, **s["kw"])
    assert fit.converged
    assert fit.loglik > s["ref"]["ll"] + 10.0
    const = fit.params.loc[fit.params["term"] == "const", "estimate"].to_numpy()
    np.testing.assert_allclose(const, [-3.0, 0.0, 3.0], atol=0.35)


def _statsmodels_eval(
    y: np.ndarray, spec: core.Spec, parts: Dict[str, np.ndarray], exog: Any = None
) -> Dict[str, Any]:
    from statsmodels.tsa.regime_switching.markov_autoregression import (
        MarkovAutoregression,
    )

    mod = MarkovAutoregression(
        y,
        k_regimes=spec.k,
        order=spec.p,
        exog=exog,
        switching_exog=True,
        switching_ar=spec.sw_ar,
        switching_variance=spec.sw_var,
    )
    vals = []
    for name in mod.param_names:
        state = int(name.split("[")[1][0]) if "[" in name else 0
        if name.startswith("p["):
            vals.append(parts["P"][int(name[2]), int(name[5])])
        elif name.startswith("const"):
            vals.append(parts["mu"][state])
        elif name.startswith("sigma2"):
            vals.append(np.exp(2.0 * parts["lnsig"][state]))
        elif name.startswith("ar.L"):
            vals.append(parts["phi"][state, int(name[4]) - 1])
        else:
            vals.append(parts["b"][state, 0])
    params = np.array(vals)
    res = mod.smooth(params)
    return {
        "ll": float(mod.loglike(params)),
        "filt": np.asarray(res.filtered_marginal_probabilities),
        "smooth": np.asarray(res.smoothed_marginal_probabilities),
    }


@pytest.mark.parametrize(
    "k,p,col,sw_ar,sw_var",
    [
        (2, 4, "y_ar", False, False),
        (2, 2, "y_ar", True, False),
        (2, 1, "y_var", False, True),
        (3, 2, "y3_ar", False, False),
        (3, 4, "y3_ar", False, False),
        (3, 1, "y3_ar", True, True),
    ],
)
def test_expanded_state_filter_matches_statsmodels(
    k: int, p: int, col: str, sw_ar: bool, sw_var: bool
) -> None:
    """statsmodels 0.14.6 ``MarkovAutoregression`` at arbitrary parameters.

    Covers the regime spaces Stata could not be made to converge on
    (three states with two or more lags). Switching variances with two or
    more lags are left to the Stata rows (``ar2var``): statsmodels and
    Stata disagree there and we match Stata.
    """
    rng = np.random.default_rng(5 + 10 * k + p)
    spec = core.Spec(k, "ar", p, 0, 0, "switch", sw_ar, sw_var)
    none = np.empty((len(DATA), 0))
    dat = core.Data(DATA[col].to_numpy(), none, none, p)
    phi = rng.uniform(-0.3, 0.4, size=(k, p))
    parts = {
        "mu": np.linspace(-2.0, 3.0, k) + 0.2 * rng.normal(size=k),
        "a": np.zeros(0),
        "b": np.zeros((k, 0)),
        "phi": phi if sw_ar else np.tile(phi[:1], (k, 1)),
        "lnsig": (
            np.log(rng.uniform(0.6, 1.2, size=k)) if sw_var else np.full(k, np.log(0.8))
        ),
        "P": 0.3 * rng.dirichlet(2.0 * np.ones(k), size=k) + 0.7 * np.eye(k),
    }
    theta = core.pack(spec, parts)
    ref = _statsmodels_eval(dat.y, spec, parts)
    fs = core.filter_smooth(spec, dat, theta)
    assert core.loglik(spec, dat, theta) == pytest.approx(ref["ll"], rel=1e-12)
    np.testing.assert_allclose(fs["filt"], ref["filt"], atol=1e-11)
    np.testing.assert_allclose(fs["smooth"], ref["smooth"], atol=1e-11)


def test_ar_switching_regressor_probabilities_follow_statsmodels() -> None:
    """The one row where Stata's -predict- and its own likelihood part ways.

    At Stata's estimates of ``mswitch ar y_z, ar(1) switch(z)`` our log
    likelihood equals Stata's ``e(ll)`` and statsmodels' to rounding, and
    our probabilities equal statsmodels'; Stata's predicted probabilities
    are up to 0.04 away from both.
    """
    s = _setup("ar1z")
    spec, dat = s["spec"], s["dat"]
    parts = {key: v[0] for key, v in core.unpack(spec, s["b"][None, :]).items()}
    ref = _statsmodels_eval(dat.y, spec, parts, exog=dat.z)
    fs = core.filter_smooth(spec, dat, s["b"])
    assert ref["ll"] == pytest.approx(s["ref"]["ll"], rel=1e-12)
    np.testing.assert_allclose(fs["filt"], ref["filt"], atol=1e-11)
    np.testing.assert_allclose(fs["smooth"], ref["smooth"], atol=1e-11)
    gap = np.abs(fs["filt"] - np.array(s["ref"]["filtered"])).max()
    assert 0.01 < gap < 0.1
