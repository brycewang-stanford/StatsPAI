"""Option and degenerate-input tests for the weak-identification tools in
``statspai.iv`` (scan round 3): ``kleibergen_paap_rk``,
``sanderson_windmeijer``, ``conditional_lr_test`` and the Anderson-Rubin /
CLR / K confidence sets.

Numerical assertions compare with algebra written here (reduced-form Wald
statistics with a stated small-sample factor, the quadratic that defines the
homoskedastic Anderson-Rubin set, the Cragg-Donald minimum eigenvalue) or
with an identity, and, for two or more endogenous regressors, with Stata 18
``ivreg2`` output quoted at the test.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statspai.exceptions import (
    IdentificationFailure,
    MethodIncompatibility,
    StatsPAIError,
)
from statspai.iv.weak_identification import (
    conditional_lr_test,
    kleibergen_paap_rk,
    sanderson_windmeijer,
)
from statspai.iv.weak_iv_ci import anderson_rubin_ci, conditional_lr_ci, k_test_ci

# closed-form comparisons on small well-conditioned designs
RTOL = 1e-9
N, K_Z = 500, 3
Z_COLS = ["z0", "z1", "z2"]


def _data(seed: int = 3, second_relevant: bool = False) -> pd.DataFrame:
    """y on d1 (and d2) with three instruments and one control. By default
    ``d2`` is unrelated to the instruments, so a model with both endogenous
    regressors is under-identified (rank of the reduced form is one)."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(N, K_Z))
    x = rng.normal(size=N)
    het = np.sqrt(1 + Z[:, 0] ** 2)
    d1 = Z @ np.array([0.6, 0.4, 0.2]) + 0.3 * x + rng.normal(size=N) * het
    d2 = rng.normal(size=N)
    if second_relevant:
        d2 = d2 + Z @ np.array([0.1, 0.5, -0.4])
    y = d1 + d2 + rng.normal(size=N) * het
    df = pd.DataFrame({"y": y, "d1": d1, "d2": d2, "x": x, "g": rng.integers(0, 25, N)})
    for j in range(K_Z):
        df[f"z{j}"] = Z[:, j]
    return df


def _partial(df, cols):
    W = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    M = df[cols].to_numpy()
    return M - W @ np.linalg.lstsq(W, M, rcond=None)[0]


def _sums(scores, groups):
    s = pd.DataFrame(scores).groupby(np.asarray(groups)).sum().to_numpy()
    return s.T @ s


# ------------------------------------------------------- Kleibergen-Paap rk


class TestKleibergenPaapOneEndogenous:
    """With one endogenous regressor the rank test is the test of Pi = 0 in
    the first stage, so both statistics are Wald / score forms that can be
    written out. K counts the constant, the control and the instruments."""

    @staticmethod
    def _pieces(df):
        D, Z = _partial(df, ["d1"])[:, 0], _partial(df, Z_COLS)
        ZZi = np.linalg.inv(Z.T @ Z)
        pi = ZZi @ Z.T @ D
        return D, Z, ZZi, pi, D - Z @ pi

    def _kp(self, df, **kw):
        return kleibergen_paap_rk(df[["d1"]], df[Z_COLS], exog=["x"], data=df, **kw)

    def test_classical_wald_f_is_the_first_stage_f(self):
        df = _data()
        D, Z, ZZi, pi, v = self._pieces(df)
        dof = N - 2 - K_Z
        f_hand = (pi @ Z.T @ Z @ pi / K_Z) / (v @ v / dof)
        res = self._kp(df, cov_type="nonrobust")
        np.testing.assert_allclose(res.rk_f, f_hand, rtol=RTOL)
        # Anderson canonical-correlation LM: n times the first-stage R^2
        np.testing.assert_allclose(res.rk_lm, N * (1 - v @ v / (D @ D)), rtol=RTOL)

    def test_robust_wald_and_lm(self):
        df = _data()
        D, Z, ZZi, pi, v = self._pieces(df)
        wald_cov = ZZi @ _sums(Z * v[:, None], np.arange(N)) @ ZZi * N / (N - 2 - K_Z)
        lm_cov = (
            ZZi @ _sums(Z * D[:, None], np.arange(N)) @ ZZi
        )  # null imposed, no factor
        res = self._kp(df, cov_type="robust")
        np.testing.assert_allclose(
            res.rk_wald, pi @ np.linalg.solve(wald_cov, pi), rtol=RTOL
        )
        np.testing.assert_allclose(
            res.rk_lm, pi @ np.linalg.solve(lm_cov, pi), rtol=RTOL
        )
        np.testing.assert_allclose(res.rk_f, res.rk_wald / K_Z, rtol=RTOL)
        np.testing.assert_allclose(
            res.rk_lm_pvalue, stats.chi2.sf(res.rk_lm, K_Z), rtol=RTOL
        )

    def test_cluster_wald_and_lm(self):
        df = _data()
        D, Z, ZZi, pi, v = self._pieces(df)
        G = df.g.nunique()
        factor = G / (G - 1) * (N - 1) / (N - 2 - K_Z)
        wald_cov = ZZi @ _sums(Z * v[:, None], df.g) @ ZZi * factor
        lm_cov = ZZi @ _sums(Z * D[:, None], df.g) @ ZZi
        res = self._kp(df, cov_type="cluster", cluster=df.g.to_numpy())
        np.testing.assert_allclose(
            res.rk_wald, pi @ np.linalg.solve(wald_cov, pi), rtol=RTOL
        )
        np.testing.assert_allclose(
            res.rk_lm, pi @ np.linalg.solve(lm_cov, pi), rtol=RTOL
        )

    def test_small_false_removes_only_the_degrees_of_freedom_factor(self):
        df = _data()
        a, b = self._kp(df), self._kp(df, small=False)
        np.testing.assert_allclose(b.rk_wald / a.rk_wald, N / (N - 2 - K_Z), rtol=RTOL)
        np.testing.assert_allclose(a.rk_lm, b.rk_lm, rtol=RTOL)

    def test_underidentified_and_bad_options_are_refused(self):
        df = _data()
        with pytest.raises(IdentificationFailure):
            kleibergen_paap_rk(df[["d1", "d2"]], df[["z0"]])
        with pytest.raises(ValueError, match="requires `cluster`"):
            self._kp(df, cov_type="cluster")
        with pytest.raises(ValueError, match="Unknown cov_type"):
            self._kp(df, cov_type="hc3")

    def test_missing_cluster_id_is_refused(self):
        df = _data()
        g = df["g"].astype(float).to_numpy().copy()
        g[:10] = np.nan
        with pytest.raises((StatsPAIError, ValueError)):
            self._kp(df, cov_type="cluster", cluster=g)

    def test_single_cluster_is_refused(self):
        df = _data()
        with pytest.raises((StatsPAIError, ValueError)):
            self._kp(df, cov_type="cluster", cluster=np.ones(N))

    @pytest.mark.parametrize("redundant", ["multiple", "constant"])
    def test_collinear_instruments_are_refused(self, redundant):
        df = _data()
        extra = 2.0 * df.z0 if redundant == "multiple" else np.ones(N)
        Z = df[Z_COLS].assign(extra=extra)
        with pytest.raises((StatsPAIError, ValueError)):
            kleibergen_paap_rk(df[["d1"]], Z, exog=["x"], data=df, cov_type="nonrobust")

    def test_cluster_without_cov_type_cluster_is_refused_or_used(self):
        df = _data()
        try:
            res = self._kp(df, cluster=df.g.to_numpy())
        except (StatsPAIError, ValueError):
            return
        clustered = self._kp(df, cov_type="cluster", cluster=df.g.to_numpy())
        np.testing.assert_allclose(res.rk_wald, clustered.rk_wald, rtol=RTOL)


class TestKleibergenPaapTwoEndogenous:
    """Two endogenous regressors: the rank statistic tests rank(Pi) = 1, not
    Pi = 0. Under classical errors the rk Wald statistic is the Cragg-Donald
    one, the smallest eigenvalue of ``S^-1/2 D'P_Z D S^-1/2'`` with ``S`` the
    reduced-form residual covariance over n - K (Stata ``ivreg2`` prints the
    same number in both places when no robust option is given)."""

    @staticmethod
    def _cragg_donald_wald(df):
        D, Z = _partial(df, ["d1", "d2"]), _partial(df, Z_COLS)
        Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T)
        V = D - Pz @ D
        S = V.T @ V / (N - 2 - K_Z)
        Li = np.linalg.inv(np.linalg.cholesky(S))
        return float(np.linalg.eigvalsh(Li @ (D.T @ Pz @ D) @ Li.T).min())

    @pytest.mark.parametrize("second_relevant", [False, True])
    def test_classical_rk_wald_is_cragg_donald(self, second_relevant):
        df = _data(second_relevant=second_relevant)
        res = kleibergen_paap_rk(
            df[["d1", "d2"]], df[Z_COLS], exog=["x"], data=df, cov_type="nonrobust"
        )
        # the eigenvalue comes from two routes of float64 algebra
        np.testing.assert_allclose(res.rk_wald, self._cragg_donald_wald(df), rtol=1e-8)

    def test_singleton_clusters_reproduce_the_robust_statistic(self):
        """G = n: G/(G-1) * (n-1)/(n-K) equals the robust n/(n-K)."""
        df = _data(second_relevant=True)
        args = (df[["d1", "d2"]], df[Z_COLS])
        robust = kleibergen_paap_rk(*args, exog=["x"], data=df, cov_type="robust")
        single = kleibergen_paap_rk(
            *args, exog=["x"], data=df, cov_type="cluster", cluster=np.arange(N)
        )
        np.testing.assert_allclose(single.rk_wald, robust.rk_wald, rtol=RTOL)

    def test_lm_does_not_reject_in_the_underidentified_design(self):
        """The LM form is a rank test: with d2 unrelated to the instruments
        it is a chi2(2) draw. 20 is far in the tail (p = 5e-5) and far below
        the Pi = 0 statistic, which exceeds 250 on this design."""
        df = _data()
        res = kleibergen_paap_rk(df[["d1", "d2"]], df[Z_COLS], exog=["x"], data=df)
        assert 0 <= res.rk_lm < 20


# ------------------------------------------------------ Sanderson-Windmeijer


class TestSandersonWindmeijer:
    def test_single_endogenous_regressor_is_the_first_stage_f(self):
        df = _data()
        D, Z = _partial(df, ["d1"])[:, 0], _partial(df, Z_COLS)
        fit = Z @ np.linalg.lstsq(Z, D, rcond=None)[0]
        f_hand = (fit @ fit / K_Z) / ((D - fit) @ (D - fit) / (N - 2 - K_Z))
        res = sanderson_windmeijer(df[["d1"]], df[Z_COLS], exog=["x"], data=df)
        np.testing.assert_allclose(res.sw_f["d1"], f_hand, rtol=RTOL)
        assert res.df_num["d1"] == K_Z and res.df_denom == N - 2 - K_Z

    def test_too_few_instruments_is_refused(self):
        df = _data()
        with pytest.raises(IdentificationFailure):
            sanderson_windmeijer(df[["d1", "d2"]], df[["z0"]])

    def test_conditional_f_uses_the_2sls_coefficient_of_the_other_regressor(self):
        df = _data()
        D, Z = _partial(df, ["d1", "d2"]), _partial(df, Z_COLS)
        Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T)
        res = sanderson_windmeijer(df[["d1", "d2"]], df[Z_COLS], exog=["x"], data=df)
        for j, name in enumerate(["d1", "d2"]):
            o = 1 - j
            delta = (D[:, o] @ Pz @ D[:, j]) / (D[:, o] @ Pz @ D[:, o])
            v = D[:, j] - delta * D[:, o]
            fit = Pz @ v
            f_hand = (fit @ fit / (K_Z - 1)) / ((v - fit) @ (v - fit) / (N - 2 - K_Z))
            np.testing.assert_allclose(res.sw_f[name], f_hand, rtol=1e-8)


# ------------------------------------------------------------------ CLR test


class TestConditionalLRTest:
    @pytest.mark.parametrize("beta0", [0.0, 0.9, 1.0])
    def test_one_instrument_clr_is_the_anderson_rubin_statistic(self, beta0):
        """Just identified: CLR, K and AR are one statistic, chi2(1)."""
        df = _data()
        res = conditional_lr_test("y", "d1", ["z0"], exog=["x"], data=df, beta0=beta0)
        np.testing.assert_allclose(res.statistic, res.ar_stat, rtol=1e-8)
        np.testing.assert_allclose(res.statistic, res.lm_stat, rtol=1e-8)
        np.testing.assert_allclose(
            res.pvalue, stats.chi2.sf(res.statistic, 1), rtol=1e-6, atol=1e-12
        )

    def test_statistic_is_bracketed_by_k_and_ar(self):
        df = _data()
        res = conditional_lr_test("y", "d1", Z_COLS, exog=["x"], data=df, beta0=0.8)
        assert res.lm_stat <= res.statistic + 1e-9 <= res.ar_stat + 1e-9

    def test_simulated_p_value_agrees_with_the_exact_one(self):
        """20,000 draws: the Monte Carlo SE of a p-value near 0.03 is
        sqrt(.03 * .97 / 20000) = 0.0012; six of those is 0.0073."""
        df = _data()
        exact = conditional_lr_test("y", "d1", Z_COLS, exog=["x"], data=df, beta0=0.8)
        sim = conditional_lr_test(
            "y", "d1", Z_COLS, exog=["x"], data=df, beta0=0.8,
            method="simulate", random_state=1,
        )  # fmt: skip
        assert sim.n_simulations == 20_000
        np.testing.assert_allclose(sim.statistic, exact.statistic, rtol=RTOL)
        assert abs(sim.pvalue - exact.pvalue) < 6 * np.sqrt(
            exact.pvalue * (1 - exact.pvalue) / 20_000
        )
        again = conditional_lr_test(
            "y", "d1", Z_COLS, exog=["x"], data=df, beta0=0.8,
            method="simulate", random_state=1,
        )  # fmt: skip
        assert again.pvalue == sim.pvalue

    def test_unknown_method_is_refused(self):
        with pytest.raises(MethodIncompatibility, match="exact"):
            conditional_lr_test("y", "d1", Z_COLS, data=_data(), method="zzz")

    def test_missing_value_does_not_produce_a_zero_p_value(self):
        df = _data()
        d = df.assign(y=df.y.where(df.index > 3))
        try:
            res = conditional_lr_test("y", "d1", Z_COLS, exog=["x"], data=d)
        except (StatsPAIError, ValueError):
            return
        assert np.isfinite(res.statistic) and res.pvalue > 0


# ------------------------------------------------------------ confidence sets


def _ar_quadratic(df, cols, level=0.95):
    """Coefficients (a, b, c) of ``a t^2 + b t + c <= 0``, the homoskedastic
    Anderson-Rubin set written as a quadratic inequality in beta0."""
    Y, D, Z = _partial(df, ["y"])[:, 0], _partial(df, ["d1"])[:, 0], _partial(df, cols)
    k = Z.shape[1]
    dof = N - 2 - k
    P = Z @ np.linalg.solve(Z.T @ Z, Z.T)
    c = stats.f.ppf(level, k, dof) * k / dof

    def q(a, b):
        return a @ P @ b - c * (a @ b - a @ P @ b)

    return q(D, D), -2 * q(Y, D), q(Y, Y)


def _weak(seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(N, K_Z))
    x, u = rng.normal(size=N), rng.normal(size=N)
    d1 = 0.02 * Z.sum(axis=1) + 0.7 * u + rng.normal(size=N)
    df = pd.DataFrame({"y": d1 + u, "d1": d1, "x": x, "g": rng.integers(0, 25, N)})
    for j in range(K_Z):
        df[f"z{j}"] = Z[:, j]
    return df


class TestAndersonRubinSet:
    @pytest.mark.parametrize("cols", [Z_COLS, ["z0"]])
    @pytest.mark.parametrize("level", [0.90, 0.95])
    def test_strong_instruments_endpoints_are_the_roots_of_the_quadratic(
        self, cols, level
    ):
        df = _data()
        a, b, c = _ar_quadratic(df, cols, level)
        assert a > 0  # bounded interval
        roots = np.sort(np.roots([a, b, c]).real)
        cs = anderson_rubin_ci("y", "d1", cols, exog=["x"], data=df, level=level)
        # the endpoints are bisected to 1e-14 relative
        np.testing.assert_allclose([cs.lower, cs.upper], roots, rtol=1e-9)
        assert cs.is_connected and not cs.is_unbounded and not cs.is_empty
        assert cs.as_intervals() == [(cs.lower, cs.upper)]

    def test_membership_agrees_with_the_statistic_at_every_grid_point(self):
        df = _data()
        cs = anderson_rubin_ci("y", "d1", Z_COLS, exog=["x"], data=df)
        a, b, c = _ar_quadratic(df, Z_COLS)
        inside = a * cs.beta_grid**2 + b * cs.beta_grid + c <= 0
        np.testing.assert_array_equal(cs.in_set, inside)
        assert inside.any() and not inside.all()

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_weak_instruments_give_an_unbounded_set_not_a_tight_one(self, seed):
        df = _weak(seed)
        a, b, c = _ar_quadratic(df, Z_COLS)
        assert a < 0  # the set is the whole line or the outside of an interval
        cs = anderson_rubin_ci("y", "d1", Z_COLS, exog=["x"], data=df)
        assert cs.is_unbounded and not cs.is_empty
        np.testing.assert_array_equal(
            cs.in_set, a * cs.beta_grid**2 + b * cs.beta_grid + c <= 0
        )
        # disconnected exactly when the quadratic has real roots
        assert cs.is_connected == (b * b - 4 * a * c < 0)

    def test_cluster_statistic_at_a_point(self):
        """Cluster-robust Wald of the instruments in ``y - b0 d`` on them,
        CR0 meat, scaled by (n - kW - k)/(n - 1) * (G - 1)/G over k."""
        df = _data()
        Y, D, Z = (
            _partial(df, ["y"])[:, 0],
            _partial(df, ["d1"])[:, 0],
            _partial(df, Z_COLS),
        )
        ZZi = np.linalg.inv(Z.T @ Z)
        G = df.g.nunique()
        grid = np.array([0.7, 0.95, 1.0, 1.3])
        cs = anderson_rubin_ci(
            "y", "d1", Z_COLS, exog=["x"], data=df, cluster="g", beta_grid=grid
        )
        for b0, got in zip(grid, cs.statistic):
            u0 = Y - b0 * D
            pi = ZZi @ Z.T @ u0
            V = ZZi @ _sums(Z * (u0 - Z @ pi)[:, None], df.g) @ ZZi
            stat = (
                pi
                @ np.linalg.solve(V, pi)
                / K_Z
                * (N - 2 - K_Z)
                / (N - 1)
                * (G - 1)
                / G
            )
            np.testing.assert_allclose(got, stat, rtol=RTOL)
        np.testing.assert_allclose(cs.critical_value, stats.f.ppf(0.95, K_Z, G - 1))
        assert cs.extra["df_denom"] == G - 1

    def test_cluster_refusals(self):
        df = _data()
        g = df["g"].astype(float).where(df.index > 5)
        with pytest.raises(MethodIncompatibility, match="missing"):
            anderson_rubin_ci("y", "d1", Z_COLS, data=df.assign(g=g), cluster="g")
        with pytest.raises(MethodIncompatibility, match="two clusters"):
            anderson_rubin_ci("y", "d1", Z_COLS, data=df.assign(g=1), cluster="g")

    def test_collinear_instruments_are_refused(self):
        df = _data()
        d = df.assign(zz=2.0 * df.z0)
        for fn in (anderson_rubin_ci, conditional_lr_ci, k_test_ci):
            with pytest.raises(IdentificationFailure, match="rank deficient"):
                fn("y", "d1", ["z0", "zz"], exog=["x"], data=d)

    @pytest.mark.parametrize("fn", [anderson_rubin_ci, conditional_lr_ci, k_test_ci])
    def test_missing_value_is_refused_or_dropped(self, fn):
        df = _data()
        d = df.assign(y=df.y.where(df.index > 3))
        try:
            cs = fn("y", "d1", Z_COLS, exog=["x"], data=d)
        except (StatsPAIError, ValueError):
            return
        pre = fn("y", "d1", Z_COLS, exog=["x"], data=d.dropna())
        np.testing.assert_allclose(
            [cs.lower, cs.upper], [pre.lower, pre.upper], rtol=1e-8
        )

    @pytest.mark.parametrize("level", [95, 1.5, 0.0])
    def test_level_outside_the_unit_interval_is_refused(self, level):
        with pytest.raises((StatsPAIError, ValueError)):
            anderson_rubin_ci("y", "d1", Z_COLS, data=_data(), level=level)

    def test_unsorted_grid_gives_the_same_set_as_the_sorted_one(self):
        df = _data()
        grid = np.array([-3.0, 0.9, 1.0, 1.1, 5.0])
        shuffled = np.array([5.0, 1.0, -3.0, 0.9, 1.1])
        a = anderson_rubin_ci("y", "d1", Z_COLS, exog=["x"], data=df, beta_grid=grid)
        b = anderson_rubin_ci(
            "y", "d1", Z_COLS, exog=["x"], data=df, beta_grid=shuffled
        )
        np.testing.assert_allclose([b.lower, b.upper], [a.lower, a.upper], rtol=1e-9)

    def test_cluster_meat_of_deficient_rank_is_refused(self):
        df = _data()
        with pytest.raises((StatsPAIError, ValueError)):
            anderson_rubin_ci(
                "y", "d1", Z_COLS, exog=["x"], data=df.assign(g=df.g % 2), cluster="g"
            )


class TestCLRAndKSets:
    def test_one_instrument_three_sets_coincide_up_to_the_reference_law(self):
        """Just identified: K and CLR are the same statistic with the same
        chi2(1) critical value, so the two sets are identical; AR uses the
        F(1, n - K) critical value and is the wider of the two by that
        margin only (ratio of critical values 1.0048 at n = 500)."""
        df = _data()
        clr = conditional_lr_ci("y", "d1", ["z0"], exog=["x"], data=df)
        k = k_test_ci("y", "d1", ["z0"], exog=["x"], data=df)
        ar = anderson_rubin_ci("y", "d1", ["z0"], exog=["x"], data=df)
        np.testing.assert_allclose(
            [clr.lower, clr.upper], [k.lower, k.upper], rtol=1e-7
        )
        assert ar.lower < k.lower < k.upper < ar.upper
        np.testing.assert_allclose([ar.lower, ar.upper], [k.lower, k.upper], rtol=5e-3)

    def test_set_contains_exactly_the_points_the_test_does_not_reject(self):
        df = _data()
        cs = conditional_lr_ci("y", "d1", Z_COLS, exog=["x"], data=df)
        for b0 in (cs.lower + 0.02, 0.5 * (cs.lower + cs.upper), cs.upper - 0.02):
            assert (
                conditional_lr_test(
                    "y", "d1", Z_COLS, exog=["x"], data=df, beta0=b0
                ).pvalue
                > 0.05
            )
        for b0 in (cs.lower - 0.02, cs.upper + 0.02):
            assert (
                conditional_lr_test(
                    "y", "d1", Z_COLS, exog=["x"], data=df, beta0=b0
                ).pvalue
                < 0.05
            )

    def test_weak_instruments_are_flagged_unbounded(self):
        df = _weak()
        for fn in (conditional_lr_ci, k_test_ci):
            assert fn("y", "d1", Z_COLS, exog=["x"], data=df).is_unbounded

    def test_unknown_method_is_refused(self):
        with pytest.raises(MethodIncompatibility, match="exact"):
            conditional_lr_ci("y", "d1", Z_COLS, data=_data(), method="zzz")


# ------------------------------------------------- Stata ivreg2, p >= 2 (pinned)


def _ivreg2_design() -> pd.DataFrame:
    """The design the Stata 18 ``ivreg2`` numbers below were computed on
    (written to CSV at full precision and read back with ``asdouble``):
    ``d2`` is unrelated to the instruments, ``d2b`` and ``d3`` are not."""
    rng = np.random.default_rng(3)
    n, k = 500, 3
    Z = rng.normal(size=(n, k))
    x = rng.normal(size=n)
    d1 = Z @ np.array([0.6, 0.4, 0.2]) + 0.3 * x + rng.normal(size=n)
    d2 = rng.normal(size=n)
    y = d1 + d2 + rng.normal(size=n)
    df = pd.DataFrame(
        {"y": y, "d1": d1, "d2": d2, "x": x, "z0": Z[:, 0], "z1": Z[:, 1]}
    )
    df["z2"] = Z[:, 2]
    df["g"] = rng.integers(0, 25, n)
    i = np.arange(1, n + 1)
    df["d2b"] = df.d2 + 0.5 * df.z1 - 0.3 * df.z2
    df["d3"] = 0.4 * df.z0 - 0.2 * df.z2 + np.sin(i)
    df["z3"] = np.cos(i * 1.7)
    return df


# ivreg2 y x (<endog> = <instruments>), [robust | cluster(g)] first
# -> e(widstat) "KP rk Wald F" (Cragg-Donald F when classical), e(idstat)
#    "KP rk LM" (Anderson LM when classical), e(idp)
_IVREG2_RK = {
    ("A", "nonrobust"): (1.63484824588, 4.90548124563, 0.0860574121076),
    ("A", "robust"): (1.52209580519, 4.55604522255, 0.102486662454),
    ("A", "cluster"): (1.15928071045, 3.44989831799, 0.1781821105),
    ("B", "nonrobust"): (41.3823504948, 100.256515142, 1.69658002526e-22),
    ("B", "robust"): (46.1760683161, 76.5130477328, 2.42886170759e-17),
    ("B", "cluster"): (35.026422408, 19.2119882114, 0.0000673239775796),
    ("C", "nonrobust"): (21.9670003164, 75.5050982994, 4.02046445165e-17),
    ("C", "robust"): (23.5664367246, 59.514247177, 1.19301369253e-13),
    ("C", "cluster"): (24.4930026304, 16.3198605212, 0.000285882331371),
}
# e(first)["SWF", .] of the classical fit
_IVREG2_SWF = {
    "A": (136.784930022, 2.4522874365),
    "B": (120.068908194, 63.1486945896),
    "C": (61.460616944, 57.987142134, 53.4184521449),
}
# e(first)["SWF", .] of ``ivreg2 y x (...), robust first`` / ``cluster(g)
# first`` (ivreg2 4.1.12, Stata 18, printed %21.15g; 2026-10-28)
_IVREG2_SWF_ROBUST = {
    ("A", "robust"): (161.55131446773, 2.28741036040931),
    ("A", "cluster"): (129.669971868042, 1.74774311263619),
    ("B", "robust"): (147.435027912521, 69.6253715262835),
    ("B", "cluster"): (140.059636548602, 53.7777059702282),
    ("C", "robust"): (74.0945794165751, 61.2097862457252, 58.4157709513561),
    ("C", "cluster"): (76.257064038924, 57.0418747487982, 53.8234024433963),
}
_IVREG2_SPECS = {
    "A": (["d1", "d2"], ["z0", "z1", "z2"]),
    "B": (["d1", "d2b"], ["z0", "z1", "z2"]),
    "C": (["d1", "d2b", "d3"], ["z0", "z1", "z2", "z3"]),
}


class TestSeveralEndogenousAgainstIvreg2:
    """Stata prints 12 significant digits; 1e-9 leaves room for the last
    printed digit and for a different LAPACK."""

    @pytest.mark.parametrize("key", sorted(_IVREG2_RK))
    def test_rk_statistics(self, key):
        spec, cov = key
        df = _ivreg2_design()
        endog, inst = _IVREG2_SPECS[spec]
        res = kleibergen_paap_rk(
            df[endog], df[inst], exog=["x"], data=df, cov_type=cov,
            cluster=df.g.to_numpy() if cov == "cluster" else None,
        )  # fmt: skip
        f, lm, p = _IVREG2_RK[key]
        np.testing.assert_allclose(res.rk_f, f, rtol=1e-9)
        np.testing.assert_allclose(res.rk_lm, lm, rtol=1e-9)
        np.testing.assert_allclose(res.rk_lm_pvalue, p, rtol=1e-7)
        assert res.df_num == len(inst)
        np.testing.assert_allclose(res.rk_wald, res.rk_f * len(inst), rtol=1e-12)
        np.testing.assert_allclose(
            res.rk_wald_pvalue,
            stats.chi2.sf(res.rk_wald, len(inst) - len(endog) + 1),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("spec", sorted(_IVREG2_SWF))
    def test_sanderson_windmeijer_f(self, spec):
        df = _ivreg2_design()
        endog, inst = _IVREG2_SPECS[spec]
        res = sanderson_windmeijer(df[endog], df[inst], exog=["x"], data=df)
        np.testing.assert_allclose(
            [res.sw_f[name] for name in endog], _IVREG2_SWF[spec], rtol=1e-9
        )
        assert res.df_denom == len(df) - 2 - len(inst)

    def test_one_endogenous_regressor_is_the_first_stage_f(self):
        """ivreg2, first on (d1 = z0 z1 z2): F = 95.2001906021."""
        df = _ivreg2_design()
        inst = ["z0", "z1", "z2"]
        sw = sanderson_windmeijer(df[["d1"]], df[inst], exog=["x"], data=df)
        kp = kleibergen_paap_rk(
            df[["d1"]], df[inst], exog=["x"], data=df, cov_type="nonrobust"
        )
        np.testing.assert_allclose(sw.sw_f["d1"], 95.2001906021, rtol=1e-9)
        np.testing.assert_allclose(kp.rk_f, 95.2001906021, rtol=1e-9)

    @pytest.mark.parametrize("cov", ["nonrobust", "robust", "cluster"])
    def test_sp_iv_reports_the_same_statistics(self, cov):
        """The diagnostics ``sp.iv`` attaches are these functions' numbers
        (the SW F used to be computed without partialling out the
        constant)."""
        import statspai as sp

        df = _ivreg2_design()
        kw = {"robust": {"robust": "robust"}, "cluster": {"cluster": "g"}}.get(cov, {})
        res = sp.iv("y ~ (d1 + d2b ~ z0 + z1 + z2) + x", df, **kw)
        f, lm, _ = _IVREG2_RK[("B", cov)]
        np.testing.assert_allclose(res.diagnostics["KP rk Wald F"], f, rtol=1e-9)
        np.testing.assert_allclose(res.diagnostics["KP rk LM"], lm, rtol=1e-9)
        # ivreg2 tests the SW first stage with the fit's variance too
        # (until 2026-10-28 sp.iv reported the classical F here)
        swf = _IVREG2_SWF["B"] if cov == "nonrobust" else _IVREG2_SWF_ROBUST[("B", cov)]
        np.testing.assert_allclose(
            [
                res.diagnostics["SW conditional F (d1)"],
                res.diagnostics["SW conditional F (d2b)"],
            ],
            swf,
            rtol=1e-9,
        )
        assert res.diagnostics["SW conditional F cov type"] == cov

    @pytest.mark.parametrize("key", sorted(_IVREG2_SWF_ROBUST))
    def test_sp_iv_robust_sw_f(self, key):
        import statspai as sp

        spec, cov = key
        endog, inst = _IVREG2_SPECS[spec]
        kw = {"robust": {"robust": "robust"}, "cluster": {"cluster": "g"}}[cov]
        df = _ivreg2_design()
        res = sp.iv(f"y ~ ({' + '.join(endog)} ~ {' + '.join(inst)}) + x", df, **kw)
        np.testing.assert_allclose(
            [res.diagnostics[f"SW conditional F ({n})"] for n in endog],
            _IVREG2_SWF_ROBUST[key],
            rtol=1e-9,
        )

    def test_sp_iv_multiway_cluster_keeps_the_labelled_classical_f(self):
        import statspai as sp

        df = _ivreg2_design().assign(h=lambda d: np.arange(len(d)) % 7)
        res = sp.iv("y ~ (d1 + d2b ~ z0 + z1 + z2) + x", df, cluster=["g", "h"])
        assert res.diagnostics["SW conditional F cov type"].startswith(
            "nonrobust (multiway"
        )
        np.testing.assert_allclose(
            res.diagnostics["SW conditional F (d1)"], _IVREG2_SWF["B"][0], rtol=1e-9
        )
