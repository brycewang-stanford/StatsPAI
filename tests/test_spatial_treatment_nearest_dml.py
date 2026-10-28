"""``sp.spatial_treatment(method='nearest' | 'dml')`` (Pollmann 2026).

The method author's example code implements only the ring (IPW)
estimator, so the two alternative estimators are checked against known
truth (T1): exact enumeration of every assignment of a small design, a
direct loop over the paper's formula, design-based Monte Carlo, and the
identity that reduces DML to the ring estimator.
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.spatial._treatments_dml import dml_ring
from statspai.spatial._treatments_nearest import (
    _closer_logprob,
    _nearest_realised,
    _nearest_variance,
)
from statspai.spatial.treatments import _ring_weights

FIX = Path(__file__).parent / "reference_parity" / "_fixtures"


def _small_design(seed=1, S=7, n=40, d0=0.35):
    rng = np.random.default_rng(seed)
    locs = rng.uniform(0, 1, (S, 2))
    units = rng.uniform(0, 1, (n, 2))
    D = np.hypot(
        units[:, None, 0] - locs[None, :, 0], units[:, None, 1] - locs[None, :, 1]
    )
    p = rng.uniform(0.2, 0.7, S)
    wk = ((D > 0) & (D <= 0.3)).astype(float) * rng.uniform(0.5, 1.5, (n, S))
    alpha = rng.normal(10, 2, n)
    return D, p, wk, alpha, d0


def _tau(d, d0):
    return 3.0 * (1.0 - d / d0)


# ---------------------------------------------------------------------
# Nearest location: exact checks by enumerating every assignment
# ---------------------------------------------------------------------


def test_nearest_variance_is_exact_up_to_the_young_bound():
    """E[V_hat] = Var(linearised estimator) + sum over incompatible pairs.

    With 7 candidates all 128 assignments are enumerated. Outcomes follow
    Assumption 4: a unit gets tau(distance to its nearest realised
    location) when that is within d0. The centring constants are fixed so
    the linearised estimator is an exact Horvitz-Thompson sum.
    """
    D, p, wk, alpha, d0 = _small_design()
    n, S = D.shape
    E = D <= d0
    log1mp = np.log1p(-p)
    logPt = _closer_logprob(D, log1mp)
    logPc = (E * log1mp).sum(1)[:, None] - E * log1mp
    Nbar = (wk * p).sum()
    mu_t, mu_c = 11.0, 9.5
    terms = []  # (kind, i, s, a_k)
    for i in range(n):
        for s in range(S):
            if wk[i, s]:
                yt = alpha[i] + (_tau(D[i, s], d0) if D[i, s] <= d0 else 0.0)
                terms.append(("t", i, s, p[s] * wk[i, s] * (yt - mu_t) / Nbar))
                terms.append(("c", i, s, -p[s] * wk[i, s] * (alpha[i] - mu_c) / Nbar))
    a = np.array([t[3] for t in terms])
    logpk = np.array(
        [
            np.log(p[s]) + logPt[i, s] if kind == "t" else log1mp[s] + logPc[i, s]
            for kind, i, s, _ in terms
        ]
    )
    allW = np.array(list(itertools.product([0, 1], repeat=S)), float)
    prW = np.prod(np.where(allW == 1, p, 1 - p), axis=1)
    ind = np.zeros((len(allW), len(terms)), bool)
    EV = 0.0
    for b, Wv in enumerate(allW):
        dmin = _nearest_realised(D, Wv)
        Nt = (Wv[None, :] == 1) & (D <= dmin[:, None])
        N0 = dmin > d0
        for k, (kind, i, s, _) in enumerate(terms):
            ind[b, k] = Nt[i, s] if kind == "t" else (Wv[s] == 0 and N0[i])
        Yobs = np.where(dmin <= d0, alpha + _tau(np.minimum(dmin, d0), d0), alpha)
        EV += (
            prW[b]
            * _nearest_variance(
                wk, D, E, p, Yobs, logPt, logPc, Nt, N0, Wv, mu_t, mu_c
            )["var"]
        )
    # Closed-form event probabilities equal their enumerated frequencies.
    np.testing.assert_allclose(prW @ ind, np.exp(logpk), atol=1e-14)
    L = ind @ (a / np.exp(logpk))
    np.testing.assert_allclose(prW @ L, a.sum(), atol=1e-12)
    var_L = prW @ (L - prW @ L) ** 2
    joint = (ind * prW[:, None]).T @ ind
    bound = np.sum((joint == 0) * (a[:, None] + a[None, :]) ** 2 / 2)
    assert bound > 0.5 * var_L  # the bound is not negligible in this design
    # Exact identity; observed relative gap 1e-15.
    assert abs(EV - var_L - bound) <= 1e-10 * var_L


def test_nearest_estimate_matches_appendix_formula_by_loop():
    rng = np.random.default_rng(4)
    S, n, d0 = 12, 120, 0.3
    locs = pd.DataFrame(rng.uniform(0, 1, (S, 2)), columns=["x", "y"])
    locs["p"] = rng.uniform(0.15, 0.6, S)
    locs["W"] = [1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    units = pd.DataFrame(rng.uniform(0, 1, (n, 2)), columns=["x", "y"])
    units["Y"] = rng.normal(size=n)
    bins = [0, 0.15, 0.3]
    res = sp.spatial_treatment(
        units,
        "Y",
        ["x", "y"],
        locs,
        treat="W",
        propensity="p",
        bins=bins,
        max_effect_distance=d0,
        n_perm=0,
        method="nearest",
    )
    D = np.hypot(
        units.x.values[:, None] - locs.x.values, units.y.values[:, None] - locs.y.values
    )
    W, p, Y = locs.W.values, locs.p.values, units.Y.values
    for k in range(2):
        lo, hi = bins[k], bins[k + 1]
        nt = dt = nc = dc = 0.0
        for s in range(S):
            for i in range(n):
                w = float(
                    (D[i, s] > lo or (k == 0 and D[i, s] == lo)) and D[i, s] <= hi
                )
                if w == 0:
                    continue
                closer = [t for t in range(S) if t != s and D[i, t] < D[i, s]]
                if W[s] == 1 and all(W[t] == 0 for t in closer):
                    pr = np.prod([1 - p[t] for t in closer])
                    nt += w * Y[i] / pr
                    dt += w / pr
                near = [t for t in range(S) if D[i, t] <= d0]
                if W[s] == 0 and all(W[t] == 0 for t in near):
                    pr = np.prod([1 - p[t] for t in near if t != s])
                    nc += p[s] / (1 - p[s]) * w * Y[i] / pr
                    dc += p[s] / (1 - p[s]) * w / pr
        assert res.table.estimate.iloc[k] == pytest.approx(nt / dt - nc / dc, abs=1e-12)
        assert res.table.mu_treated.iloc[k] == pytest.approx(nt / dt, abs=1e-12)


def test_nearest_recovers_effect_under_assumption_4():
    """Design-based Monte Carlo: fixed units, redrawn assignments."""
    rng = np.random.default_rng(7)
    k = 8
    g = (np.arange(k) + 0.5) / k
    locs = pd.DataFrame({"x": np.repeat(g, k), "y": np.tile(g, k)})
    locs["p"] = rng.uniform(0.15, 0.35, len(locs))
    units = pd.DataFrame(rng.uniform(0, 1, (800, 2)), columns=["x", "y"])
    D = np.hypot(
        units.x.values[:, None] - locs.x.values, units.y.values[:, None] - locs.y.values
    )
    d0, bins = 0.12, [0, 0.06, 0.12]
    alpha = 10 + 3 * np.sin(5 * units.x.values) + rng.normal(size=len(units))
    p = locs.p.values
    # Estimand: tau(d) averaged over ring pairs with weight p_s.
    truth = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        w = (D >= lo) & (D <= hi) if lo == 0 else (D > lo) & (D <= hi)
        truth.append(float((w * p * _tau(D, d0)).sum() / (w * p).sum()))
    est, se = [], []
    for _ in range(200):
        W = (rng.uniform(size=len(locs)) < p).astype(int)
        dmin = np.where(W == 1, D, np.inf).min(axis=1)
        y = alpha + np.where(dmin <= d0, _tau(np.minimum(dmin, d0), d0), 0.0)
        r = sp.spatial_treatment(
            units.assign(Y=y),
            "Y",
            ["x", "y"],
            locs.assign(W=W),
            treat="W",
            propensity="p",
            bins=bins,
            max_effect_distance=d0,
            n_perm=0,
            method="nearest",
        )
        est.append(r.table.estimate.values)
        se.append(r.table.se.values)
    est, se = np.array(est), np.array(se)
    mc_sd = est.std(axis=0)
    # Ratio estimator: bias is O(1/n); 4 Monte Carlo SEs of the mean.
    np.testing.assert_array_less(
        np.abs(est.mean(axis=0) - truth), 4 * mc_sd / np.sqrt(len(est))
    )
    # Conservative standard error: at least the Monte Carlo SD on average
    # (observed mean SE / MC SD = 1.05 and 1.31).
    assert np.all(se.mean(axis=0) >= 0.95 * mc_sd)
    cover = (np.abs(est - truth) <= 1.96 * se).mean(axis=0)
    assert np.all(cover >= 0.90)  # observed 0.935 / 0.985 over 200 draws


def test_nearest_randomization_pvalue_recomputes_permuted_estimates():
    rng = np.random.default_rng(2)
    locs = pd.DataFrame(rng.uniform(0, 1, (30, 2)), columns=["x", "y"])
    locs["realized"] = (np.arange(30) % 3 == 0).astype(int)
    units = pd.DataFrame(rng.uniform(0, 1, (300, 2)), columns=["x", "y"])
    units["Y"] = rng.normal(size=300)
    kw = dict(bins=[0, 0.1, 0.2], max_effect_distance=0.2, method="nearest")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.spatial_treatment(
            units, "Y", ["x", "y"], locs, n_perm=40, seed=9, **kw
        )
        perm_rng = np.random.default_rng(9)
        draws = [perm_rng.permutation(locs.realized.values) for _ in range(40)]
        stats_b = np.array(
            [
                sp.spatial_treatment(
                    units, "Y", ["x", "y"], locs.assign(realized=Wb), n_perm=0, **kw
                ).table.estimate.values
                for Wb in draws
            ]
        )
    for k in range(2):
        est = res.table.estimate.iloc[k]
        ok = np.isfinite(stats_b[:, k])
        hits = np.sum(np.abs(stats_b[ok, k]) >= abs(est) - 1e-12 * max(1, abs(est)))
        assert res.table.ri_pvalue.iloc[k] == pytest.approx((1 + hits) / (1 + ok.sum()))
    assert res.diagnostics["randomization"] == "permutation"
    assert {"V_cov", "V_bound"} <= set(res.table.columns)
    assert np.allclose(res.table.se**2, res.table.V_cov + res.table.V_bound)


# ---------------------------------------------------------------------
# DML
# ---------------------------------------------------------------------


@pytest.mark.parametrize("design,d0", [(1, 0.15), (2, 0.10)])
def test_dml_with_control_mean_as_first_step_is_the_ring_estimator(design, d0):
    """m = inverse-odds control mean everywhere gives Equation 3 exactly.

    The control sum of ``Y - mu_c`` is then zero, the treated part is
    ``mu_t - mu_c``, and the linearisation (treated centred at the
    estimate, control at zero) is term by term the ring estimator's.
    """
    units = pd.read_csv(FIX / "spatial_treatment_units.csv")
    locs = pd.read_csv(FIX / "spatial_treatment_locations.csv")
    u = units[units.design == design].reset_index(drop=True)
    loc = locs[locs.design == design].reset_index(drop=True)
    bins = [0, 0.05, 0.10, 0.15]
    ipw = sp.spatial_treatment(
        u,
        "Y",
        ["x", "y"],
        loc,
        treat="W",
        propensity="p",
        bins=bins,
        max_effect_distance=d0,
        n_perm=0,
    ).table
    D = np.hypot(u.x.values[:, None] - loc.x.values, u.y.values[:, None] - loc.y.values)
    E = sparse.csr_matrix(D <= d0)
    W, p = loc.W.values.astype(float), loc.p.values
    for k in range(3):
        wk = _ring_weights(D, bins[k], bins[k + 1], "bin", first=(k == 0))
        fit = dml_ring(wk, E, W, p, u.Y.values, np.full(D.shape, ipw.mu_control[k]))
        assert fit["est"] == pytest.approx(ipw.estimate[k], abs=1e-12)
        assert np.sqrt(fit["comp"]["var"]) == pytest.approx(ipw.se[k], rel=1e-12)
        for key in ("Vt", "Vc", "Vx"):
            assert fit["comp"][key] == pytest.approx(
                ipw[key.replace("V", "V_")][k], rel=1e-10, abs=1e-15
            )


def _confounded_design():
    rng = np.random.default_rng(0)
    k = 10
    g = (np.arange(k) + 0.5) / k
    locs = pd.DataFrame({"x": np.repeat(g, k), "y": np.tile(g, k)})
    locs["p"] = 1 / (1 + np.exp(-(-1.5 + 3 * locs.x)))
    units = pd.DataFrame(rng.uniform(0, 1, (1000, 2)), columns=["x", "y"])
    D = np.hypot(
        units.x.values[:, None] - locs.x.values, units.y.values[:, None] - locs.y.values
    )
    return rng, locs, units, D


def test_dml_recovers_effect_when_locations_are_confounded():
    """Candidates east are more likely realised and units east have higher
    outcomes. The ring estimator with the realised share is biased; DML
    with a cross-fitted logit on the location's x and the unit's x in the
    outcome model is not. Effect: +2 per realised location within 0.04."""
    rng, locs, units, D = _confounded_design()
    alpha = 10 + 4 * units.x.values
    bins, d0 = [0, 0.04, 0.08], 0.08
    dml, naive, se = [], [], []
    for b in range(100):
        W = (rng.uniform(size=len(locs)) < locs.p).astype(int)
        y = alpha + 2 * (D[:, W == 1] <= 0.04).sum(1) + rng.normal(size=len(units))
        u, L = units.assign(Y=y), locs.assign(realized=W)
        r = sp.spatial_treatment(
            u,
            "Y",
            ["x", "y"],
            L,
            bins=bins,
            max_effect_distance=d0,
            n_perm=0,
            seed=b,
            method="dml",
            covariates=["x"],
            location_covariates=["x"],
        )
        dml.append(r.table.estimate.values)
        se.append(r.table.se.values)
        naive.append(
            sp.spatial_treatment(
                u, "Y", ["x", "y"], L, bins=bins, max_effect_distance=d0, n_perm=0
            ).table.estimate.values
        )
    dml, naive, se = np.array(dml), np.array(naive), np.array(se)
    truth = np.array([2.0, 0.0])
    mc_sd = dml.std(axis=0)
    assert np.all(np.abs(naive.mean(axis=0) - truth) > 0.5)  # confounding is real
    np.testing.assert_array_less(
        np.abs(dml.mean(axis=0) - truth), 4 * mc_sd / np.sqrt(len(dml))
    )
    # Design-based SE: conservative, not wildly so (observed ratio 1.5 / 3).
    ratio = se.mean(axis=0) / mc_sd
    assert np.all(ratio >= 0.9) and np.all(ratio <= 4.0)
    assert r.diagnostics["propensity_source"] == "cross-fitted logit"
    assert r.model_info["estimator"] == "dml"
    assert np.all(np.isnan(r.table.ri_pvalue))


def test_dml_known_propensity_and_bookkeeping():
    rng, locs, units, D = _confounded_design()
    W = (rng.uniform(size=len(locs)) < locs.p).astype(int)
    units["Y"] = 10 + 2 * (D[:, W == 1] <= 0.04).sum(1) + rng.normal(size=len(units))
    units["region"] = np.where(units.x > 0.5, "east", "west")
    r = sp.spatial_treatment(
        units,
        "Y",
        ["x", "y"],
        locs.assign(realized=W),
        propensity="p",
        bins=[0, 0.04, 0.08],
        max_effect_distance=0.08,
        method="dml",
        covariates=["region"],  # categorical: expanded to an indicator
        n_folds=8,
    )
    assert np.array_equal(r.diagnostics["propensity_hat"], locs.p.values)
    assert r.diagnostics["propensity_source"] == "p"
    assert r.diagnostics["randomization"] is None
    assert "randomization_note" in r.diagnostics
    assert len(r.diagnostics["n_train_units"]) == 8
    t = r.table
    np.testing.assert_allclose(t.estimate, t.mu_treated - t.mu_control, atol=1e-12)
    assert np.allclose(t.se**2, (t.V_t + t.V_c + t.V_x) / len(locs))
    # Same seed, same folds, same numbers.
    r2 = sp.spatial_treatment(
        units,
        "Y",
        ["x", "y"],
        locs.assign(realized=W),
        propensity="p",
        bins=[0, 0.04, 0.08],
        max_effect_distance=0.08,
        method="dml",
        covariates=["region"],
        n_folds=8,
    )
    assert r.table.equals(r2.table)


def test_explicit_ipw_is_the_default():
    rng, locs, units, D = _confounded_design()
    W = (rng.uniform(size=len(locs)) < locs.p).astype(int)
    units["Y"] = rng.normal(size=len(units))
    kw = dict(bins=[0, 0.04, 0.08], n_perm=19, seed=1)
    a = sp.spatial_treatment(units, "Y", ["x", "y"], locs.assign(realized=W), **kw)
    b = sp.spatial_treatment(
        units, "Y", ["x", "y"], locs.assign(realized=W), method="ipw", **kw
    )
    assert a.table.equals(b.table)
    assert a.model_info["estimator"] == "ipw"


def test_alternative_method_inputs():
    rng, locs, units, D = _confounded_design()
    locs["realized"] = (rng.uniform(size=len(locs)) < 0.3).astype(int)
    units["Y"] = rng.normal(size=len(units))
    base = dict(data=units, y="Y", coords=["x", "y"], locations=locs, n_perm=0)
    bins = [0, 0.04, 0.08]
    with pytest.raises(MethodIncompatibility, match="method must be"):
        sp.spatial_treatment(**base, bins=bins, method="ols")
    with pytest.raises(MethodIncompatibility, match="method='dml' only"):
        sp.spatial_treatment(**base, bins=bins, covariates=["x"])
    with pytest.raises(MethodIncompatibility, match="not both"):
        sp.spatial_treatment(
            **base,
            bins=bins,
            method="dml",
            propensity="p",
            location_covariates=["x"],
        )
    with pytest.raises(MethodIncompatibility, match="n_folds"):
        sp.spatial_treatment(**base, bins=bins, method="dml", n_folds=1)
    with pytest.raises(DataInsufficient, match="no column"):
        sp.spatial_treatment(**base, bins=bins, method="dml", covariates=["zz"])
    # Rings and exposure radius so wide that no unit is far from a fold.
    with pytest.raises(DataInsufficient, match="Fold"):
        sp.spatial_treatment(
            **base, bins=[0, 0.5, 1.0], method="dml", n_folds=2, seed=0
        )
