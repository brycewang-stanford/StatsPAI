"""Tests for ``sp.spatial_treatment`` (Pollmann 2026, design-based)."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.spatial.treatments import _ring_variance


def _grid_design(seed=0, n_units=600, k=8, p=0.4, d_eff=0.04):
    rng = np.random.default_rng(seed)
    g = (np.arange(k) + 0.5) / k
    locs = pd.DataFrame({"x": np.repeat(g, k), "y": np.tile(g, k)})
    locs["p"] = p
    units = pd.DataFrame(rng.uniform(0, 1, (n_units, 2)), columns=["x", "y"])
    d = np.hypot(
        units.x.values[:, None] - locs.x.values, units.y.values[:, None] - locs.y.values
    )
    base = 10 + 3 * np.sin(5 * units.x.values) + rng.normal(size=n_units)
    tau_is = 2.0 * (d <= d_eff)  # additive, local effects
    return locs, units, base, tau_is


# ---------------------------------------------------------------------
# T1: known effect profile, design-based Monte Carlo
# ---------------------------------------------------------------------


def test_recovers_effect_profile_and_coverage():
    locs, units, base, tau_is = _grid_design()
    rng = np.random.default_rng(1)
    est, se = [], []
    for _ in range(200):
        W = (rng.uniform(size=len(locs)) < locs.p.values).astype(int)
        u = units.copy()
        u["Y"] = base + tau_is @ W  # potential outcomes fixed; only W varies
        lc = locs.assign(realized=W)
        r = sp.spatial_treatment(
            u, "Y", ["x", "y"], lc, propensity="p", bins=[0, 0.04, 0.08], n_perm=0
        )
        est.append(r.table.estimate.to_numpy())
        se.append(r.table.se.to_numpy())
    est, se = np.array(est), np.array(se)
    truth = np.array([2.0, 0.0])  # ATT under additivity: 2 within 0.04, 0 beyond
    mc_sd = est.std(axis=0)
    # Bias within 3 Monte Carlo standard errors of the mean.
    assert np.all(np.abs(est.mean(axis=0) - truth) < 3 * mc_sd / np.sqrt(200))
    cover = (np.abs(est - truth) <= 1.96 * se).mean(axis=0)
    # Conservative design-based SE: coverage at least about nominal
    # (0.90 allows 200-draw noise around 0.95+).
    assert np.all(cover >= 0.90)
    # The SE should not be wildly conservative either.
    ratio = (se**2).mean(axis=0) / mc_sd**2
    assert np.all((ratio > 0.9) & (ratio < 2.5))


# ---------------------------------------------------------------------
# Exact enumeration of the design
# ---------------------------------------------------------------------


def _enumerate(locs_xy, units_xy, p, base, tau_is, edges, d0):
    """Exact design moments of tau_tilde and of the variance estimator."""
    d = np.hypot(
        units_xy[:, None, 0] - locs_xy[None, :, 0],
        units_xy[:, None, 1] - locs_xy[None, :, 1],
    )
    S = len(p)
    w = ((d > edges[0]) & (d <= edges[1])).astype(float)
    E = sparse.csr_matrix(d <= d0)
    nbar = float(w.sum(axis=0) @ p) / S
    r = p / (1 - p)
    assigns = [np.array(a, float) for a in itertools.product([0, 1], repeat=S)]
    prob = np.array([np.prod(np.where(a == 1, p, 1 - p)) for a in assigns])
    Ys = [base + tau_is @ a for a in assigns]
    num_t = sum(
        pr * (w * a[None, :] * Y[:, None]).sum() for pr, a, Y in zip(prob, assigns, Ys)
    )
    den_t = sum(pr * (w * a[None, :]).sum() for pr, a in zip(prob, assigns))
    num_c = sum(
        pr * (w * ((1 - a) * r)[None, :] * Y[:, None]).sum()
        for pr, a, Y in zip(prob, assigns, Ys)
    )
    mu_t, mu_c = num_t / den_t, num_c / (S * nbar)
    Dvals, Vhat = [], []
    for a, Y in zip(assigns, Ys):
        Dn = (w * a[None, :] * (Y - mu_t)[:, None]).sum() - (
            w * ((1 - a) * r)[None, :] * (Y - mu_c)[:, None]
        ).sum()
        Dvals.append(Dn / (S * nbar))
        Vhat.append(
            _ring_variance(sparse.csr_matrix(w), E, a, p, Y, mu_t, mu_c, nbar, S)["var"]
        )
    Dvals, Vhat = np.array(Dvals), np.array(Vhat)
    return float(prob @ Dvals), float(prob @ Dvals**2), float(prob @ Vhat)


def test_variance_estimator_is_conservative_in_expectation():
    # Overlapping exposures: candidates close enough that units see several.
    rng = np.random.default_rng(3)
    locs_xy = rng.uniform(0, 1, (7, 2))
    units_xy = rng.uniform(0, 1, (60, 2))
    p = rng.uniform(0.2, 0.7, 7)
    d = np.hypot(
        units_xy[:, None, 0] - locs_xy[None, :, 0],
        units_xy[:, None, 1] - locs_xy[None, :, 1],
    )
    base = rng.normal(5, 2, 60)
    tau_is = rng.normal(1.5, 1.0, (60, 7)) * (d <= 0.3)  # heterogeneous, local
    mean_D, var_tilde, mean_vhat = _enumerate(
        locs_xy, units_xy, p, base, tau_is, (0.0, 0.3), 0.3
    )
    assert abs(mean_D) < 1e-12  # tau_tilde is exactly unbiased
    assert mean_vhat >= var_tilde * (1 - 1e-10)
    assert np.unique((d <= 0.3).sum(axis=1)).size > 1  # exposures overlap


def test_variance_estimator_is_exact_when_bound_is_tight():
    # Clusters far apart, constant effect, cluster-constant outcomes: the
    # dropped terms vanish and E[V_hat] equals var(tau_tilde) exactly.
    locs_xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    p = np.array([0.3, 0.5, 0.6, 0.4, 0.45])
    rng = np.random.default_rng(5)
    sizes = [3, 5, 2, 4, 6]
    units_xy = np.vstack(
        [locs_xy[s] + rng.uniform(-0.03, 0.03, (n, 2)) for s, n in enumerate(sizes)]
    )
    cl = np.repeat(np.arange(5), sizes)
    base = np.array([1.0, 4.0, -2.0, 0.5, 3.0])[cl]
    d = np.hypot(
        units_xy[:, None, 0] - locs_xy[None, :, 0],
        units_xy[:, None, 1] - locs_xy[None, :, 1],
    )
    tau_is = 1.7 * (d <= 0.1)
    mean_D, var_tilde, mean_vhat = _enumerate(
        locs_xy, units_xy, p, base, tau_is, (0.0, 0.1), 0.1
    )
    assert abs(mean_D) < 1e-12
    np.testing.assert_allclose(mean_vhat, var_tilde, rtol=1e-10)


def test_clustered_variance_matches_hand_derivation():
    # One location per cluster, every unit exposed to its own location only:
    # S^2 nbar^2 V_hat = sum_treated [(1-p) a^2 + p n sum e^2]
    #                  + sum_control [p r^2 a^2 + r p n sum e^2].
    rng = np.random.default_rng(7)
    S, sizes = 6, np.array([2, 3, 4, 1, 5, 3])
    p = rng.uniform(0.2, 0.7, S)
    W = np.array([1, 0, 1, 0, 0, 1], float)
    cl = np.repeat(np.arange(S), sizes)
    n = sizes.sum()
    Y = rng.normal(size=n)
    w = np.zeros((n, S))
    w[np.arange(n), cl] = 1.0
    E = sparse.csr_matrix(w > 0)
    mu_t, mu_c, nbar = 0.3, -0.2, float(w.sum(0) @ p) / S
    got = _ring_variance(sparse.csr_matrix(w), E, W, p, Y, mu_t, mu_c, nbar, S)["var"]
    r = p / (1 - p)
    tot = 0.0
    for s in range(S):
        e = Y[cl == s] - (mu_t if W[s] else mu_c)
        a2, se2, ns = e.sum() ** 2, (e**2).sum(), sizes[s]
        if W[s]:
            tot += (1 - p[s]) * a2 + p[s] * ns * se2
        else:
            tot += p[s] * r[s] ** 2 * a2 + r[s] * p[s] * ns * se2
    np.testing.assert_allclose(got, tot / (S * nbar) ** 2, rtol=1e-12)


# ---------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------


def test_equal_propensities_give_pair_weighted_difference_in_means():
    locs, units, base, tau_is = _grid_design(seed=4, k=6)
    W = np.random.default_rng(9).binomial(1, 0.5, len(locs))
    units["Y"] = base + tau_is @ W
    r = sp.spatial_treatment(
        units, "Y", ["x", "y"], locs.assign(realized=W), bins=[0, 0.06], n_perm=0
    )
    d = np.hypot(
        units.x.values[:, None] - locs.x.values, units.y.values[:, None] - locs.y.values
    )
    inring = d <= 0.06
    t_mean = units.Y.values @ (inring[:, W == 1]).sum(1) / inring[:, W == 1].sum()
    c_mean = units.Y.values @ (inring[:, W == 0]).sum(1) / inring[:, W == 0].sum()
    assert r.estimate == pytest.approx(t_mean - c_mean, rel=1e-12)
    assert r.diagnostics["propensity_source"] == "realised share"


def test_location_weights_average_location_means():
    locs, units, base, tau_is = _grid_design(seed=6, k=5)
    W = np.array([1, 0] * 12 + [1])
    units["Y"] = base + tau_is @ W
    r = sp.spatial_treatment(
        units,
        "Y",
        ["x", "y"],
        locs.assign(realized=W),
        bins=[0, 0.1],
        location_weights=True,
        n_perm=0,
    )
    d = np.hypot(
        units.x.values[:, None] - locs.x.values, units.y.values[:, None] - locs.y.values
    )
    means = np.array([units.Y.values[d[:, s] <= 0.1].mean() for s in range(len(locs))])
    has = np.array([(d[:, s] <= 0.1).any() for s in range(len(locs))])
    want = means[has & (W == 1)].mean() - means[has & (W == 0)].mean()
    assert r.estimate == pytest.approx(want, rel=1e-12)


def test_precomputed_distances_and_haversine():
    locs, units, base, tau_is = _grid_design(seed=8, k=5)
    W = np.random.default_rng(2).binomial(1, 0.5, len(locs))
    units["Y"] = base + tau_is @ W
    lc = locs.assign(realized=W)
    a = sp.spatial_treatment(units, "Y", ["x", "y"], lc, bins=[0, 0.05, 0.1], n_perm=0)
    D = np.hypot(
        units.x.values[:, None] - locs.x.values, units.y.values[:, None] - locs.y.values
    )
    b = sp.spatial_treatment(
        units, "Y", locations=lc, distances=D, bins=[0, 0.05, 0.1], n_perm=0
    )
    pd.testing.assert_frame_equal(a.table, b.table)
    # One degree of latitude on the mean-radius sphere.
    one = pd.DataFrame({"lat": [0.0], "lon": [0.0]})
    from statspai.spatial.treatments import _distances

    dd = _distances(one.values, np.array([[1.0, 0.0]]), "haversine", "km")
    assert dd[0, 0] == pytest.approx(np.pi / 180 * 6371.0088, rel=1e-12)


def test_randomization_pvalue():
    locs, units, base, tau_is = _grid_design(seed=10)
    W = np.random.default_rng(4).binomial(1, 0.4, len(locs))
    units["Y"] = base + 25 * (tau_is @ W)  # huge effect within 0.04
    lc = locs.assign(realized=W)
    r1 = sp.spatial_treatment(units, "Y", ["x", "y"], lc, bins=[0, 0.04], n_perm=199)
    r2 = sp.spatial_treatment(units, "Y", ["x", "y"], lc, bins=[0, 0.04], n_perm=199)
    assert r1.table.ri_pvalue.iloc[0] == pytest.approx(1 / 200)
    assert r1.table.ri_pvalue.iloc[0] == r2.table.ri_pvalue.iloc[0]  # seeded
    assert r1.diagnostics["randomization"] == "permutation"
    r3 = sp.spatial_treatment(
        units, "Y", ["x", "y"], lc, bins=[0, 0.04], propensity="p", n_perm=99
    )
    assert r3.diagnostics["randomization"] == "bernoulli"


def test_triangular_kernel_and_result_surface():
    locs, units, base, tau_is = _grid_design(seed=12, k=6)
    W = np.random.default_rng(1).binomial(1, 0.5, len(locs))
    units["Y"] = base + tau_is @ W
    r = sp.spatial_treatment(
        units,
        "Y",
        ["x", "y"],
        locs.assign(realized=W),
        bins=[0, 0.04, 0.08],
        kernel="triangular",
        n_perm=0,
    )
    assert np.all(np.isfinite(r.table.se)) and np.all(r.table.se > 0)
    out = r.to_dict(detail="agent")
    assert out["estimate"] == pytest.approx(r.table.estimate.iloc[0])
    assert "Spatial treatment" in r.summary()
    import matplotlib

    matplotlib.use("Agg")
    fig, ax = r.plot()
    assert len(ax.lines) >= 1


# ---------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------


def _small():
    locs, units, base, tau_is = _grid_design(seed=13, k=4, n_units=200)
    W = np.array([1, 0] * 8)
    units["Y"] = base + tau_is @ W
    return locs.assign(realized=W), units


def test_empty_ring_warns_and_is_missing():
    lc, u = _small()
    with pytest.warns(UserWarning, match="no units"):
        r = sp.spatial_treatment(u, "Y", ["x", "y"], lc, bins=[0, 0.1, 5, 6], n_perm=0)
    assert np.isnan(r.table.estimate.iloc[2]) and r.diagnostics["empty_rings"] == [2]
    assert np.isfinite(r.estimate)


def test_no_estimable_ring_raises():
    lc, u = _small()
    with pytest.raises(DataInsufficient):
        sp.spatial_treatment(u, "Y", ["x", "y"], lc, bins=[5, 6], n_perm=0)


@pytest.mark.parametrize("realized", [0, 1])
def test_no_realised_or_no_counterfactual(realized):
    lc, u = _small()
    with pytest.raises(DataInsufficient):
        sp.spatial_treatment(
            u, "Y", ["x", "y"], lc.assign(realized=realized), bins=[0, 0.1], n_perm=0
        )


def test_bad_inputs():
    lc, u = _small()
    bad = u.copy()
    bad.loc[0, "x"] = np.nan
    with pytest.raises(DataInsufficient):
        sp.spatial_treatment(bad, "Y", ["x", "y"], lc, bins=[0, 0.1])
    with pytest.raises(MethodIncompatibility):
        sp.spatial_treatment(u.assign(x="a"), "Y", ["x", "y"], lc, bins=[0, 0.1])
    with pytest.raises(DataInsufficient):
        sp.spatial_treatment(u, "Y", ["x", "z"], lc, bins=[0, 0.1])
    with pytest.raises(MethodIncompatibility):
        sp.spatial_treatment(u, "Y", ["x", "y"], lc, bins=[0.1, 0.05])
    with pytest.raises(MethodIncompatibility):
        sp.spatial_treatment(
            u, "Y", ["x", "y"], lc.assign(p=1.0), propensity="p", bins=[0, 0.1]
        )
    with pytest.raises(MethodIncompatibility):
        sp.spatial_treatment(
            u, "Y", locations=lc, distances=np.ones((3, 3)), bins=[0, 0.1]
        )
    with pytest.raises(DataInsufficient):
        sp.spatial_treatment(u.assign(Y=np.nan), "Y", ["x", "y"], lc, bins=[0, 0.1])
    with pytest.raises(MethodIncompatibility):
        sp.spatial_treatment(u, "Y", ["x", "y"], lc.assign(realized=2), bins=[0, 0.1])
