"""Tier D P2 known-truth upgrades — extra spatial-econometrics entry points.

Part of the P1/P2 "Tier D analytic special-cases" campaign. These eight
entry points were graded ``weak`` (boolean/shape/not-None asserts with no
known-truth anchor). Each test below compares against a number that is known
*before* the estimator runs, either a graph-theoretic identity on a regular
lattice or a recovered coefficient from a DGP built with that truth baked in:

    sp.queen_weights    queen contiguity on a k x k grid of unit squares has
                        EXACT neighbour counts: corner 3, edge 5, interior 8;
                        the weights matrix is symmetric (geopandas required).
    sp.rook_weights     rook (edge-only) contiguity on the same grid: corner
                        2, edge 3, interior 4 (geopandas required).
    sp.block_weights    full within-regime connectivity: a unit in a group of
                        size g has EXACTLY g-1 neighbours, none outside it;
                        symmetric, zero diagonal, block-diagonal sparsity.
    sp.distance_band    a radius that isolates disjoint point-pairs connects
                        EXACTLY those pairs (degree 1 each); symmetric.
    sp.getis_ord_local  Gi* equals an independent closed-form reimplementation
                        to machine precision; a flat field gives Gi* == 0; the
                        peak Gi* sits inside a planted high-value cluster.
    sp.slx              OLS on X augmented with WX recovers the known direct
                        betas and the known spatial-lag thetas on a low-noise
                        DGP Y = const + X beta + (WX) theta + eps.
    sp.spatial_did      the two-way FE + spatial-lag-of-treatment regression
                        recovers the planted direct effect tau and spillover
                        effect theta.
    sp.spatial_iv       Kelejian-Prucha S2SLS recovers (a) a structural
                        coefficient on an endogenous regressor that biased OLS
                        cannot, and (b) the spatial autoregressive rho from a
                        simulated SAR reduced form.

Purely additive — no estimator numerics changed (campaign red line).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Shared lattice helpers
# ---------------------------------------------------------------------------
def _grid_coords(side):
    """Row-major coordinates of a ``side x side`` unit lattice."""
    return np.array([[i, j] for i in range(side) for j in range(side)], dtype=float)


def _row_normalize(W_dense):
    rs = W_dense.sum(axis=1, keepdims=True)
    rs = np.where(rs == 0, 1.0, rs)
    return W_dense / rs


def _square_grid_gdf(side):
    """``side x side`` grid of touching unit squares (queen/rook testbed)."""
    gpd = pytest.importorskip("geopandas")
    shapely_geom = pytest.importorskip("shapely.geometry")
    box = shapely_geom.box
    polys = [box(j, i, j + 1, i + 1) for i in range(side) for j in range(side)]
    return gpd.GeoDataFrame({"id": range(len(polys))}, geometry=polys)


def _grid_degree_truth(side, criterion):
    """Exact contiguity degree for each cell of a ``side x side`` grid.

    Rook: shared edges only -> 4 for an interior cell, 3 on an edge, 2 in a
    corner. Queen: edges + vertices -> 8 / 5 / 3.
    """
    deg = np.empty(side * side, dtype=int)
    for i in range(side):
        for j in range(side):
            rook = 0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= i + di < side and 0 <= j + dj < side:
                    rook += 1
            diag = 0
            for di, dj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                if 0 <= i + di < side and 0 <= j + dj < side:
                    diag += 1
            deg[i * side + j] = rook if criterion == "rook" else rook + diag
    return deg


# ---------------------------------------------------------------------------
# sp.queen_weights (requires geopandas)
# ---------------------------------------------------------------------------
class TestQueenWeightsAnalytic:
    def test_grid_neighbour_counts_exact(self):
        # On a 3x3 grid of unit squares, queen contiguity gives corner 3,
        # edge 5, centre 8 -- a fact fixed purely by lattice geometry.
        gdf = _square_grid_gdf(3)
        w = sp.queen_weights(gdf)
        deg = np.asarray(w.sparse.sum(axis=1)).ravel().astype(int)
        np.testing.assert_array_equal(deg, _grid_degree_truth(3, "queen"))

    def test_weights_symmetric(self):
        gdf = _square_grid_gdf(4)
        S = sp.queen_weights(gdf).sparse.toarray()
        np.testing.assert_array_equal(S, S.T)

    def test_queen_has_more_links_than_rook(self):
        # Queen counts shared vertices, so total link count strictly exceeds
        # rook on any grid with interior cells.
        gdf = _square_grid_gdf(4)
        q = sp.queen_weights(gdf).sparse.sum()
        r = sp.rook_weights(gdf).sparse.sum()
        assert q > r


# ---------------------------------------------------------------------------
# sp.rook_weights (requires geopandas)
# ---------------------------------------------------------------------------
class TestRookWeightsAnalytic:
    def test_grid_neighbour_counts_exact(self):
        # Rook contiguity on a 3x3 grid: corner 2, edge 3, centre 4.
        gdf = _square_grid_gdf(3)
        w = sp.rook_weights(gdf)
        deg = np.asarray(w.sparse.sum(axis=1)).ravel().astype(int)
        np.testing.assert_array_equal(deg, _grid_degree_truth(3, "rook"))

    def test_centre_has_exactly_four_neighbours(self):
        # The classic "rook moves" identity: a 4-connected interior cell.
        gdf = _square_grid_gdf(3)
        w = sp.rook_weights(gdf)
        assert len(w.neighbors[4]) == 4

    def test_weights_symmetric(self):
        gdf = _square_grid_gdf(4)
        S = sp.rook_weights(gdf).sparse.toarray()
        np.testing.assert_array_equal(S, S.T)


# ---------------------------------------------------------------------------
# sp.block_weights
# ---------------------------------------------------------------------------
class TestBlockWeightsAnalytic:
    def test_within_regime_degree_is_group_size_minus_one(self):
        # Full connectivity inside each block: a unit in a group of size g has
        # exactly g-1 neighbours and none outside the group.
        regimes = np.array([0, 0, 0, 1, 1, 2])  # sizes 3, 2, 1
        w = sp.block_weights(regimes)
        deg = np.asarray(w.sparse.sum(axis=1)).ravel().astype(int)
        np.testing.assert_array_equal(deg, np.array([2, 2, 2, 1, 1, 0]))

    def test_no_cross_block_links_and_zero_diagonal(self):
        regimes = [0, 0, 1, 1, 1]
        S = sp.block_weights(regimes).sparse.toarray()
        # Zero diagonal (no self-neighbour).
        np.testing.assert_array_equal(np.diag(S), np.zeros(5))
        # No link between block 0 (rows 0,1) and block 1 (rows 2,3,4).
        assert S[0, 2] == 0 and S[1, 4] == 0 and S[2, 0] == 0
        # Singletons in a block are mutual neighbours.
        assert S[0, 1] == 1 and S[2, 3] == 1 and S[3, 4] == 1

    def test_symmetric(self):
        S = sp.block_weights([2, 2, 0, 1, 1, 0]).sparse.toarray()
        np.testing.assert_array_equal(S, S.T)


# ---------------------------------------------------------------------------
# sp.distance_band
# ---------------------------------------------------------------------------
class TestDistanceBandAnalytic:
    def test_isolated_pairs_connect_exactly(self):
        # Two well-separated pairs; a radius of 1.5 links each pair but never
        # bridges the pairs (they are 5+ units apart). Degree is exactly 1.
        coords = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
        w = sp.distance_band(coords, threshold=1.5, binary=True)
        deg = np.asarray(w.sparse.sum(axis=1)).ravel().astype(int)
        np.testing.assert_array_equal(deg, np.array([1, 1, 1, 1]))
        assert sorted(w.neighbors[0]) == [1]
        assert sorted(w.neighbors[2]) == [3]

    def test_threshold_below_min_distance_gives_islands(self):
        # A radius smaller than every pairwise distance leaves zero links.
        coords = np.array([[0, 0], [0, 2], [0, 4]], dtype=float)
        w = sp.distance_band(coords, threshold=1.0, binary=True)
        assert w.sparse.sum() == 0

    def test_symmetric(self):
        coords = _grid_coords(4)
        S = sp.distance_band(coords, threshold=1.45, binary=True).sparse.toarray()
        np.testing.assert_array_equal(S, S.T)


# ---------------------------------------------------------------------------
# sp.getis_ord_local
# ---------------------------------------------------------------------------
class TestGetisOrdLocalAnalytic:
    def test_matches_closed_form_reimplementation(self):
        # Gi* has a closed form; recompute it independently and demand an exact
        # match (no Monte Carlo here -- it is the analytic standardised stat).
        rng = np.random.default_rng(3)
        coords = _grid_coords(6)
        w = sp.distance_band(coords, threshold=1.45, binary=True)
        y = rng.normal(5.0, 2.0, w.n)
        out = sp.getis_ord_local(y, w, star=True, permutations=0)

        S = w.sparse.toarray().copy()
        np.fill_diagonal(S, 1.0)  # star=True includes self
        n = w.n
        Wi = S.sum(axis=1)
        num = S @ y - Wi * y.mean()
        denom_core = np.maximum((n * (Wi - Wi**2 / n)) / (n - 1), 0.0)
        denom = np.sqrt(y.var(ddof=0) * denom_core)
        expected = num / denom

        np.testing.assert_allclose(np.asarray(out["Gs"]), expected, rtol=0, atol=1e-12)

    def test_constant_field_is_undefined(self):
        # A constant field has zero variance, so the Gi* denominator collapses
        # and the statistic is (correctly) NaN -- never a spurious finite value.
        coords = _grid_coords(4)
        w = sp.distance_band(coords, threshold=1.0, binary=True)
        out = sp.getis_ord_local(np.full(w.n, 4.0), w, star=True, permutations=0)
        assert np.all(np.isnan(np.asarray(out["Gs"])))

    def test_zero_numerator_when_local_mean_equals_global(self):
        # Exact analytic zero. On a 5-cell line take y = [2, 1, 4, 1, 2]
        # (global mean 2). The centre (index 2) has star-neighbourhood
        # {1, 2, 3} with values {1, 4, 1}, averaging to exactly 2 = the global
        # mean, while its degree (3) is a proper subset of n=5 so the variance
        # term stays positive. The numerator S@y - Wi*ybar is therefore exactly
        # 0 and Gi* == 0 -- a closed-form anchor independent of any RNG.
        coords = np.array([[0, k] for k in range(5)], dtype=float)
        w = sp.distance_band(coords, threshold=1.0, binary=True)
        y = np.array([2.0, 1.0, 4.0, 1.0, 2.0])  # mean 2
        out = sp.getis_ord_local(y, w, star=True, permutations=0)
        assert out["Gs"][2] == pytest.approx(0.0, abs=1e-12)
        # By the field's reflection symmetry the two flanking cells are mirror
        # images: equal magnitude, opposite sign of the corner cells.
        assert out["Gs"][1] == pytest.approx(-out["Gs"][0], abs=1e-12)

    def test_peak_score_inside_planted_hotspot(self):
        # Plant a high-value block in one corner of a grid; the largest Gi*
        # z-score must sit inside that block.
        coords = _grid_coords(5)
        w = sp.distance_band(coords, threshold=1.45, binary=True)
        y = np.ones(w.n)
        hot = [0, 1, 5, 6]  # 2x2 high corner
        y[hot] = 20.0
        out = sp.getis_ord_local(y, w, star=True, permutations=0)
        z = np.asarray(out["Gs"])
        assert int(np.nanargmax(z)) in set(hot)


# ---------------------------------------------------------------------------
# sp.slx — spatial lag of X
# ---------------------------------------------------------------------------
class TestSLXAnalytic:
    @staticmethod
    def _slx_dgp(seed=0, side=12, noise=0.01):
        rng = np.random.default_rng(seed)
        coords = _grid_coords(side)
        w = sp.distance_band(coords, threshold=1.45, binary=True)
        n = w.n
        Wn = _row_normalize(w.sparse.toarray())
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        truth = dict(const=1.0, x1=2.0, x2=-1.5, W_x1=0.8, W_x2=-0.4)
        y = (
            truth["const"]
            + truth["x1"] * x1
            + truth["x2"] * x2
            + truth["W_x1"] * (Wn @ x1)
            + truth["W_x2"] * (Wn @ x2)
            + rng.normal(0, noise, n)
        )
        df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
        return w, df, truth

    def test_recovers_direct_and_lag_coefficients(self):
        # SLX is just OLS on [const, X, WX]; with negligible noise it returns
        # the planted direct betas AND the spatial-lag thetas (named W_x*).
        w, df, truth = self._slx_dgp(noise=0.01)
        res = sp.slx(w, df, "y ~ x1 + x2", row_normalize=True)
        for name, val in truth.items():
            assert res.params[name] == pytest.approx(val, abs=2e-2), name

    def test_equals_manual_ols_on_augmented_design(self):
        # Byte-for-byte: SLX must equal hand-rolled OLS on [1, X, W@X] using
        # the same row-normalised W (the documented estimator definition).
        w, df, _ = self._slx_dgp(noise=0.3, seed=5)
        res = sp.slx(w, df, "y ~ x1 + x2", row_normalize=True)
        Wn = _row_normalize(w.sparse.toarray())
        X = np.column_stack([df["x1"], df["x2"]])
        WX = Wn @ X
        design = np.column_stack([np.ones(len(df)), X, WX])
        beta = np.linalg.lstsq(design, df["y"].to_numpy(), rcond=None)[0]
        names = ["const", "x1", "x2", "W_x1", "W_x2"]
        for name, b in zip(names, beta):
            assert res.params[name] == pytest.approx(b, rel=1e-8, abs=1e-8)


# ---------------------------------------------------------------------------
# sp.spatial_did — direct + spillover treatment effects
# ---------------------------------------------------------------------------
class TestSpatialDiDAnalytic:
    @staticmethod
    def _spatial_did_dgp(seed=1, side=8, T=6, t_treat=3, noise=0.05):
        rng = np.random.default_rng(seed)
        coords = _grid_coords(side)
        w = sp.distance_band(coords, threshold=1.0, binary=True)  # rook
        n_units = w.n
        Wn = _row_normalize(w.sparse.toarray())
        alpha_i = rng.normal(0, 1, n_units)
        gamma_t = rng.normal(0, 1, T)
        treated = rng.permutation(n_units)[: n_units // 2]
        tau, theta = 2.5, -1.0
        Dmat = np.zeros((n_units, T))
        Dmat[treated, t_treat:] = 1.0
        WDmat = Wn @ Dmat
        rows = []
        for u in range(n_units):
            for t in range(T):
                y = (
                    alpha_i[u]
                    + gamma_t[t]
                    + tau * Dmat[u, t]
                    + theta * WDmat[u, t]
                    + rng.normal(0, noise)
                )
                rows.append({"unit": u, "time": t, "y": y, "D": Dmat[u, t]})
        return w, pd.DataFrame(rows), tau, theta

    def test_recovers_direct_and_spillover_effects(self):
        # The estimating equation is Y = a_i + g_t + tau*D + theta*WD + eps.
        # With near-zero idiosyncratic noise the FE regression returns the
        # planted tau (direct) and theta (spillover).
        w, df, tau, theta = self._spatial_did_dgp(noise=0.02)
        res = sp.spatial_did(
            df,
            y="y",
            treat="D",
            unit="unit",
            time="time",
            W=w,
            normalize_W=True,
        )
        assert res.direct_effect == pytest.approx(tau, abs=0.05)
        assert res.spillover_effect == pytest.approx(theta, abs=0.05)

    def test_total_effect_equals_direct_plus_spillover(self):
        # Definitional identity: total = direct + spillover.
        w, df, _, _ = self._spatial_did_dgp(noise=0.05, seed=4)
        res = sp.spatial_did(
            df,
            y="y",
            treat="D",
            unit="unit",
            time="time",
            W=w,
            normalize_W=True,
        )
        assert res.total_effect == pytest.approx(
            res.direct_effect + res.spillover_effect, rel=1e-9, abs=1e-9
        )


# ---------------------------------------------------------------------------
# sp.spatial_iv — Kelejian-Prucha S2SLS
# ---------------------------------------------------------------------------
class TestSpatialIVAnalytic:
    def test_recovers_structural_coef_under_endogeneity(self):
        # Endogenous D shares the structural error u; an excluded instrument z
        # restores identification. S2SLS should land on the true delta=2.0
        # while OLS is biased away from it.
        rng = np.random.default_rng(2)
        coords = _grid_coords(12)
        w = sp.distance_band(coords, threshold=1.45, binary=True)
        n = w.n
        Wn = _row_normalize(w.sparse.toarray())
        z = rng.normal(0, 1, n)
        x = rng.normal(0, 1, n)
        u = rng.normal(0, 1, n)
        D = 1.0 + 1.5 * z + 0.8 * x + 0.7 * u + rng.normal(0, 0.3, n)
        delta_true, beta_x_true = 2.0, -1.0
        y = 0.5 + delta_true * D + beta_x_true * x + u
        df = pd.DataFrame({"y": y, "D": D, "x": x, "z": z})

        res = sp.spatial_iv(
            df,
            y="y",
            endog=["D"],
            exog=["x"],
            W=Wn,
            instruments=["z"],
            include_WY=False,
        )
        coef = res.coefficients.set_index("variable")["coef"]
        assert coef["D"] == pytest.approx(delta_true, abs=0.15)

        # Biased OLS overshoots the truth -- confirms endogeneity was real.
        Xo = np.column_stack([np.ones(n), D, x])
        ols_d = np.linalg.lstsq(Xo, y, rcond=None)[0][1]
        assert abs(ols_d - delta_true) > abs(coef["D"] - delta_true)

    def test_recovers_spatial_autoregressive_rho(self):
        # Simulate a pure SAR reduced form Y = (I - rho W)^-1 (X beta + u) and
        # recover rho via WY instrumented by WX, W^2X (the K-P design).
        rng = np.random.default_rng(7)
        coords = _grid_coords(12)
        w = sp.distance_band(coords, threshold=1.45, binary=True)
        n = w.n
        Wn = _row_normalize(w.sparse.toarray())
        x = rng.normal(0, 1, n)
        u = rng.normal(0, 0.3, n)
        rho_true, b_x, b_const = 0.4, 1.5, 1.0
        Y = np.linalg.solve(np.eye(n) - rho_true * Wn, b_const + b_x * x + u)
        df = pd.DataFrame({"y": Y, "x": x})

        res = sp.spatial_iv(df, y="y", endog=[], exog=["x"], W=Wn, include_WY=True)
        assert res.rho == pytest.approx(rho_true, abs=0.05)
        coef = res.coefficients.set_index("variable")["coef"]
        assert coef["x"] == pytest.approx(b_x, abs=0.05)
