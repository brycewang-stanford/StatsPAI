"""Bootstrap, jackknife and wild cluster bootstrap: options and edge cases.

Each stochastic procedure is compared with a loop written here that uses
the same generator and the same draws, or with a full enumeration; nothing
asserts that one seed lands near a truth.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.inference.iv_wild import iv_wild_bootstrap


def _clustered(seed: int = 3, n_clusters: int = 12, size: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(n_clusters), size)
    u = rng.normal(size=n_clusters)[g]
    n = g.size
    return pd.DataFrame(
        {"y": np.exp(u + 0.5 * rng.normal(size=n)), "g": g, "x": rng.normal(size=n)}
    )


def _mean(d: pd.DataFrame) -> float:
    return float(d["y"].mean())


# --------------------------------------------------------------------- #
#  sp.bootstrap
# --------------------------------------------------------------------- #


class TestBootstrap:
    def test_iid_draws_are_the_seeded_generator(self):
        df = _clustered()
        n = len(df)
        y = df["y"].to_numpy()
        rs = np.random.RandomState(5)
        draws = np.array(
            [y[rs.choice(n, size=n, replace=True)].mean() for _ in range(200)]
        )
        res = sp.bootstrap(df, _mean, n_boot=200, seed=5, alpha=0.1)
        np.testing.assert_allclose(res.boot_distribution, draws, rtol=1e-12)
        assert res.se == pytest.approx(draws.std(ddof=1), rel=1e-12)
        lo, hi = np.percentile(draws, [5, 95])
        assert (res.ci_lower, res.ci_upper) == pytest.approx((lo, hi), abs=1e-12)
        assert res.n_boot == 200

    def test_cluster_draws_keep_clusters_whole(self):
        df = _clustered()
        y = df["y"].to_numpy()
        ids = df["g"].unique()
        rows = {c: np.flatnonzero(df["g"].to_numpy() == c) for c in ids}
        rs = np.random.RandomState(8)
        draws = []
        for _ in range(100):
            pick = rs.choice(ids, size=len(ids), replace=True)
            draws.append(y[np.concatenate([rows[c] for c in pick])].mean())
        res = sp.bootstrap(df, _mean, n_boot=100, cluster="g", seed=8)
        np.testing.assert_allclose(res.boot_distribution, draws, rtol=1e-12)
        # block= resamples the same groups with the same draws
        blk = sp.bootstrap(df, _mean, n_boot=100, block="g", seed=8)
        np.testing.assert_allclose(blk.boot_distribution, draws, rtol=1e-12)

    def test_interval_types_are_their_definitions_on_the_draws(self):
        df = _clustered()
        base = sp.bootstrap(df, _mean, n_boot=300, seed=2)
        draws, est = base.boot_distribution, base.estimate
        normal = sp.bootstrap(df, _mean, n_boot=300, seed=2, ci_method="normal")
        z = stats.norm.ppf(0.975)
        assert normal.ci_lower == pytest.approx(est - z * draws.std(ddof=1))
        assert normal.ci_upper == pytest.approx(est + z * draws.std(ddof=1))

        # BCa by hand: bias correction from the draws, acceleration from the
        # delete-one jackknife (60 rows, below the 200-unit cap).
        y = df["y"].to_numpy()
        z0 = stats.norm.ppf(np.mean(draws < est))
        jack = np.array([np.delete(y, i).mean() for i in range(len(y))])
        dev = jack.mean() - jack
        acc = np.sum(dev**3) / (6 * np.sum(dev**2) ** 1.5)
        ends = []
        for q in (0.025, 0.975):
            zq = stats.norm.ppf(q)
            ends.append(stats.norm.cdf(z0 + (z0 + zq) / (1 - acc * (z0 + zq))))
        expected = np.percentile(draws, 100 * np.array(ends))
        bca = sp.bootstrap(df, _mean, n_boot=300, seed=2, ci_method="bca")
        assert (bca.ci_lower, bca.ci_upper) == pytest.approx(tuple(expected), abs=1e-10)

    def test_p_value_counts_centred_draws_beyond_the_distance_to_the_null(self):
        df = _clustered()
        res = sp.bootstrap(df, _mean, n_boot=200, seed=4, null_value=2.5)
        dist = np.abs(res.boot_distribution - res.estimate)
        expected = max(np.mean(dist >= abs(res.estimate - 2.5)), 1 / 201)
        assert res.pvalue == pytest.approx(expected, abs=1e-12)
        at_estimate = sp.bootstrap(
            df, _mean, n_boot=200, seed=4, null_value=res.estimate
        )
        assert at_estimate.pvalue == 1.0

    def test_failed_replicates_are_excluded_and_reported(self):
        df = _clustered()
        calls = [0]

        def flaky(d):
            calls[0] += 1
            if calls[0] % 3 == 0:
                raise ValueError("no fit")
            return float(d["y"].mean())

        with pytest.warns(RuntimeWarning, match="20/60 replications failed"):
            res = sp.bootstrap(df, flaky, n_boot=60, seed=1)
        # call 1 is the full sample; calls 3, 6, ..., 60 fail
        assert res.n_boot == 40
        assert res.boot_distribution.size == 40
        assert res.pvalue >= 1 / 41

    def test_too_few_surviving_replicates_is_an_error(self):
        df = _clustered()
        with pytest.raises(RuntimeError, match="at least 10"):
            sp.bootstrap(df, _mean, n_boot=5, seed=1)

    def test_invalid_options(self):
        df = _clustered()
        with pytest.raises(MethodIncompatibility, match="Unknown ci_method"):
            sp.bootstrap(df, _mean, n_boot=20, seed=1, ci_method="basic")
        with pytest.raises(MethodIncompatibility, match="idcluster"):
            sp.bootstrap(df, _mean, n_boot=20, idcluster="newid")
        with pytest.raises(MethodIncompatibility, match="estimate, standard error"):
            sp.bootstrap(df, _mean, n_boot=20, ci_method="studentized")

    def test_vector_statistic_covariance_is_the_covariance_of_the_draws(self):
        df = _clustered()
        n = len(df)
        rs = np.random.RandomState(6)
        draws = np.array(
            [
                df[["y", "x"]].to_numpy()[rs.choice(n, size=n, replace=True)].mean(0)
                for _ in range(80)
            ]
        )
        res = sp.bootstrap(df, lambda d: d[["y", "x"]].mean(), n_boot=80, seed=6)
        np.testing.assert_allclose(
            res.std_errors.to_numpy(), draws.std(axis=0, ddof=1), rtol=1e-10
        )
        np.testing.assert_allclose(
            res.model_info["bias"].to_numpy(),
            draws.mean(0) - df[["y", "x"]].mean().to_numpy(),
            atol=1e-12,
        )
        assert res.model_info["n_boot"] == 80

    def test_bca_with_block_uses_the_resampling_unit(self):
        df = _clustered()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = sp.bootstrap(
                df, _mean, n_boot=300, cluster="g", ci_method="bca", seed=1
            )
            b = sp.bootstrap(df, _mean, n_boot=300, block="g", ci_method="bca", seed=1)
        # the two options draw the same resamples (see the test above)
        assert (b.ci_lower, b.ci_upper) == pytest.approx(
            (a.ci_lower, a.ci_upper), abs=1e-10
        )

    def test_bca_does_not_depend_on_the_index_labels(self):
        df = _clustered()
        relabelled = df.copy()
        relabelled.index = np.repeat(np.arange(len(df) // 2), 2)
        a = sp.bootstrap(df, _mean, n_boot=300, ci_method="bca", seed=1)
        b = sp.bootstrap(relabelled, _mean, n_boot=300, ci_method="bca", seed=1)
        assert (b.ci_lower, b.ci_upper) == pytest.approx(
            (a.ci_lower, a.ci_upper), abs=1e-10
        )

    def test_a_single_cluster_is_refused(self):
        df = _clustered().assign(g=1)
        with pytest.raises((ValueError, RuntimeError)):
            sp.bootstrap(df, _mean, n_boot=50, cluster="g", seed=1)

    def test_missing_cluster_ids_are_refused_with_a_message(self):
        df = _clustered()
        df = df.assign(g=df["g"].astype(float).where(df.index >= 5))
        with pytest.raises(ValueError, match="(?i)missing|NaN"):
            sp.bootstrap(df, _mean, n_boot=50, cluster="g", seed=1)

    def test_missing_block_ids_are_refused(self):
        df = _clustered()
        df = df.assign(g=df["g"].astype(float).where(df.index >= 5))
        with pytest.raises(ValueError, match="(?i)missing|NaN"):
            sp.bootstrap(df, _mean, n_boot=50, block="g", seed=1)

    def test_unknown_ci_method_is_refused_for_a_vector_statistic(self):
        df = _clustered()
        with pytest.raises(MethodIncompatibility):
            sp.bootstrap(
                df,
                lambda d: d[["y", "x"]].mean(),
                n_boot=30,
                seed=1,
                ci_method="basic",
            )

    def test_alpha_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValueError):
            sp.bootstrap(_clustered(), _mean, n_boot=50, seed=1, alpha=2)


# --------------------------------------------------------------------- #
#  sp.jackknife
# --------------------------------------------------------------------- #


class TestJackknife:
    def test_mean_has_the_closed_form_standard_error(self):
        df = _clustered()
        n = len(df)
        res = sp.jackknife(df, _mean, alpha=0.1)
        se = df["y"].std(ddof=1) / np.sqrt(n)
        assert res.se == pytest.approx(se, rel=1e-12)
        assert res.bias == pytest.approx(0.0, abs=1e-10)
        crit = stats.t.ppf(0.95, n - 1)
        assert res.ci_upper == pytest.approx(res.estimate + crit * se, rel=1e-12)
        assert res.pvalue == pytest.approx(
            2 * stats.t.sf(res.estimate / se, n - 1), rel=1e-10
        )
        assert res.n_reps == n

    def test_bias_of_the_plug_in_variance_is_recovered_exactly(self):
        # The jackknife bias estimate of the divisor-n variance is exactly
        # minus the unbiased variance over n.
        df = _clustered()
        n = len(df)
        res = sp.jackknife(df, lambda d: float(d["y"].var(ddof=0)))
        assert res.bias == pytest.approx(-df["y"].var(ddof=1) / n, rel=1e-9)

    def test_cluster_and_mse_options(self):
        df = _clustered()
        y, g = df["y"].to_numpy(), df["g"].to_numpy()
        loo = np.array([y[g != c].mean() for c in np.unique(g)])
        G = loo.size
        var = (G - 1) / G * np.sum((loo - loo.mean()) ** 2)
        res = sp.jackknife(df, _mean, cluster="g")
        assert res.se == pytest.approx(np.sqrt(var), rel=1e-12)
        assert res.n_reps == G
        # equal cluster sizes, so the mean of the replicates is the estimate
        # and the documented gap (n - 1) * (centre difference)^2 is zero ...
        mse = sp.jackknife(df, _mean, cluster="g", mse=True)
        assert mse.se == pytest.approx(res.se, rel=1e-10)

        # ... and for a ratio it is not
        def ratio(d):
            return float(d["y"].mean() / (2 + d["x"].mean()))

        a = sp.jackknife(df, ratio)
        b = sp.jackknife(df, ratio, mse=True)
        n = a.n_reps
        gap = (n - 1) * (a.replicates.mean() - a.estimate) ** 2
        assert b.se**2 == pytest.approx(a.se**2 + gap, rel=1e-10)

    def test_regression_vector_is_the_leave_one_out_covariance(self):
        df = _clustered()
        n = len(df)
        X = np.column_stack([np.ones(n), df["x"]])
        y = df["y"].to_numpy()
        betas = np.array(
            [
                np.linalg.lstsq(np.delete(X, i, 0), np.delete(y, i), rcond=None)[0]
                for i in range(n)
            ]
        )
        dev = betas - betas.mean(0)
        cov = (n - 1) / n * dev.T @ dev
        fit = sp.jackknife(df, lambda d: sp.regress("y ~ x", data=d))
        np.testing.assert_allclose(
            fit.std_errors.to_numpy(), np.sqrt(np.diag(cov)), rtol=1e-9
        )
        assert fit.data_info["df_resid"] == n - 1

    def test_failed_replicates_shrink_n_and_are_reported(self):
        df = _clustered()

        def picky(d):
            if 3 not in d.index:  # leaving row 3 out fails
                raise ValueError("no")
            return float(d["y"].mean())

        with pytest.warns(RuntimeWarning, match="1 of 60"):
            res = sp.jackknife(df, picky)
        y = df["y"].to_numpy()
        loo = np.array([np.delete(y, i).mean() for i in range(60) if i != 3])
        assert res.n_reps == 59
        assert res.se == pytest.approx(
            np.sqrt(58 / 59 * np.sum((loo - loo.mean()) ** 2)), rel=1e-12
        )

    def test_degenerate_inputs_are_refused(self):
        df = _clustered()
        with pytest.raises(DataInsufficient, match="only 2"):
            sp.jackknife(df.iloc[:2], _mean)
        with pytest.raises(DataInsufficient, match="only 1 clusters"):
            sp.jackknife(df.assign(g=1), _mean, cluster="g")
        with pytest.raises(MethodIncompatibility, match="missing values"):
            sp.jackknife(
                df.assign(g=df["g"].astype(float).where(df.index > 4)),
                _mean,
                cluster="g",
            )
        with pytest.raises(MethodIncompatibility, match="not found in data"):
            sp.jackknife(df, _mean, cluster="nope")
        with pytest.raises(MethodIncompatibility, match="function"):
            sp.jackknife(df, "mean")

    def test_alpha_outside_the_unit_interval_is_refused(self):
        with pytest.raises(ValueError):
            sp.jackknife(_clustered(), _mean, alpha=1.5)


# --------------------------------------------------------------------- #
#  jackknife_se / cr2_se on a fitted regression
# --------------------------------------------------------------------- #


def _reg_frame(seed: int = 3, n: int = 200, n_clusters: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_clusters, n)
    x1 = rng.normal(size=n) + rng.normal(size=n_clusters)[g]
    x2 = rng.normal(size=n)
    y = (
        1
        + 0.5 * x1
        - 0.3 * x2
        + 0.4 * x1 * x2
        + rng.normal(size=n) * (1 + np.abs(x1))
        + rng.normal(size=n_clusters)[g]
    )
    return pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "g": g, "w": rng.uniform(0.5, 3, n)}
    )


def _cluster_jackknife(X, y, cl, w=None):
    w = np.ones(len(y)) if w is None else w
    betas = []
    for c in np.unique(cl):
        m = cl != c
        root = np.sqrt(w[m])
        betas.append(np.linalg.lstsq(X[m] * root[:, None], y[m] * root, rcond=None)[0])
    betas = np.array(betas)
    dev = betas - betas.mean(0)
    G = len(betas)
    return np.sqrt(np.diag((G - 1) / G * dev.T @ dev))


class TestJackknifeSe:
    def test_additive_formula_is_the_delete_one_cluster_variance(self):
        df = _reg_frame()
        X = np.column_stack([np.ones(len(df)), df["x1"], df["x2"]])
        fit = sp.regress("y ~ x1 + x2", data=df, cluster="g")
        jk = sp.jackknife_se(fit, df, "g")
        expected = _cluster_jackknife(X, df["y"].to_numpy(), df["g"].to_numpy())
        np.testing.assert_allclose(jk.std_errors.to_numpy(), expected, rtol=1e-9)
        assert jk.data_info["df_resid"] == df["g"].nunique() - 1
        np.testing.assert_allclose(jk.params.to_numpy(), fit.params.to_numpy())

    def test_rows_dropped_by_the_fit_are_dropped_here(self):
        df = _reg_frame()
        holes = df.assign(x2=df["x2"].where(df.index >= 10))
        fit = sp.regress("y ~ x1 + x2", data=holes, cluster="g")
        jk = sp.jackknife_se(fit, holes, "g")
        kept = df.iloc[10:]
        X = np.column_stack([np.ones(len(kept)), kept["x1"], kept["x2"]])
        expected = _cluster_jackknife(X, kept["y"].to_numpy(), kept["g"].to_numpy())
        np.testing.assert_allclose(jk.std_errors.to_numpy(), expected, rtol=1e-9)

    @pytest.mark.parametrize("formula", ["y ~ x1 + x2 + x1:x2", "y ~ x1 * x2"])
    def test_interaction_terms_are_part_of_the_refits(self, formula):
        df = _reg_frame()
        X = np.column_stack([np.ones(len(df)), df["x1"], df["x2"], df["x1"] * df["x2"]])
        fit = sp.regress(formula, data=df, cluster="g")
        jk = sp.jackknife_se(fit, df, "g")
        expected = _cluster_jackknife(X, df["y"].to_numpy(), df["g"].to_numpy())
        got = jk.std_errors[["Intercept", "x1", "x2", "x1:x2"]].to_numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-8)

    def test_weighted_fit_is_refitted_with_its_weights(self):
        df = _reg_frame()
        X = np.column_stack([np.ones(len(df)), df["x1"], df["x2"]])
        fit = sp.regress("y ~ x1 + x2", data=df, cluster="g", weights="w")
        jk = sp.jackknife_se(fit, df, "g")
        expected = _cluster_jackknife(
            X, df["y"].to_numpy(), df["g"].to_numpy(), df["w"].to_numpy()
        )
        np.testing.assert_allclose(jk.std_errors.to_numpy(), expected, rtol=1e-8)

    def test_no_intercept_fit_is_refitted_without_one(self):
        df = _reg_frame()
        X = np.column_stack([df["x1"], df["x2"]])
        fit = sp.regress("y ~ x1 + x2 - 1", data=df, cluster="g")
        jk = sp.jackknife_se(fit, df, "g")
        expected = _cluster_jackknife(X, df["y"].to_numpy(), df["g"].to_numpy())
        np.testing.assert_allclose(
            jk.std_errors[["x1", "x2"]].to_numpy(), expected, rtol=1e-8
        )

    def test_a_single_cluster_is_refused(self):
        df = _reg_frame()
        fit = sp.regress("y ~ x1 + x2", data=df, cluster="g")
        with pytest.raises(ValueError):
            sp.jackknife_se(fit, df.assign(g=0), "g")


class TestCr2:
    def test_singleton_clusters_give_hc2(self):
        df = _reg_frame()
        fit = sp.regress("y ~ x1 + x2", data=df, cluster="g")
        cr2 = sp.cr2_se(fit, df.assign(g=np.arange(len(df))), "g")
        hc2 = sp.regress("y ~ x1 + x2", data=df, robust="hc2")
        # (I - H_ii)^(-1/2) e_i squared is e_i^2 / (1 - h_ii)
        np.testing.assert_allclose(
            cr2.std_errors.to_numpy(), hc2.std_errors.to_numpy(), rtol=1e-9
        )

    def test_cr2_is_the_adjusted_residual_sandwich(self):
        df = _reg_frame()
        X = np.column_stack([np.ones(len(df)), df["x1"], df["x2"], df["x1"] * df["x2"]])
        y, cl = df["y"].to_numpy(), df["g"].to_numpy()
        bread = np.linalg.inv(X.T @ X)
        e = y - X @ bread @ X.T @ y
        meat = np.zeros((4, 4))
        for c in np.unique(cl):
            m = cl == c
            H = X[m] @ bread @ X[m].T
            vals, vecs = np.linalg.eigh(np.eye(m.sum()) - H)
            adj = vecs @ np.diag(vals**-0.5) @ vecs.T
            s = X[m].T @ adj @ e[m]
            meat += np.outer(s, s)
        expected = np.sqrt(np.diag(bread @ meat @ bread))
        fit = sp.regress("y ~ x1 * x2", data=df, cluster="g")
        cr2 = sp.cr2_se(fit, df, "g")
        got = cr2.std_errors[["Intercept", "x1", "x2", "x1:x2"]].to_numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-9)
        dof = cr2.diagnostics["satterthwaite_dof"]
        assert all(1.0 < v <= df["g"].nunique() for v in dof.values())

    def test_a_single_cluster_is_refused(self):
        df = _reg_frame()
        fit = sp.regress("y ~ x1 + x2", data=df, cluster="g")
        with pytest.raises(ValueError):
            sp.cr2_se(fit, df.assign(g=0), "g")


# --------------------------------------------------------------------- #
#  wild cluster bootstrap
# --------------------------------------------------------------------- #


def _wild_frame(seed: int = 0, n_clusters: int = 7, size: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cl = np.repeat(np.arange(n_clusters), size)
    n = cl.size
    x = rng.normal(size=n) + rng.normal(size=n_clusters)[cl]
    z = rng.normal(size=n)
    y = 1 + 0.3 * x + 0.5 * z + rng.normal(size=n) + rng.normal(size=n_clusters)[cl]
    sub = cl * 2 + (np.arange(n) % 2)
    return pd.DataFrame({"y": y, "x": x, "z": z, "cl": cl, "sub": sub})


def _cr1_t(X, y, err_idx, j, h0):
    """t statistic of coefficient j with the CR1 variance over ``err_idx``."""
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    b = bread @ X.T @ y
    e = y - X @ b
    G = int(err_idx.max()) + 1
    scores = np.zeros((G, k))
    np.add.at(scores, err_idx, X * e[:, None])
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * bread @ scores.T @ scores @ bread
    return b[j], np.sqrt(V[j, j]), (b[j] - h0) / np.sqrt(V[j, j])


def _wcr_by_refitting(df, h0, W, boot_col="cl"):
    """Restricted wild bootstrap t statistics, one OLS refit per draw."""
    y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df["x"], df["z"]])
    err = df["cl"].to_numpy()
    boot = pd.factorize(df[boot_col], sort=True)[0]
    j = 1
    others = [0, 2]
    br = np.linalg.lstsq(X[:, others], y - h0 * X[:, j], rcond=None)[0]
    fitted = X[:, others] @ br + h0 * X[:, j]
    resid = y - fitted
    beta, se, t_obs = _cr1_t(X, y, err, j, h0)
    t_star = np.array([_cr1_t(X, fitted + w[boot] * resid, err, j, h0)[2] for w in W])
    return beta, se, t_obs, t_star


def _strict_p(t_star, t_obs, W):
    """#{|t*| > |t|} / B, never counting the all +1 / all -1 draws."""
    tie = np.all(W == 1.0, axis=1) | np.all(W == -1.0, axis=1)
    return float(np.mean((np.abs(t_star) > abs(t_obs)) & ~tie))


class TestWildClusterBootstrap:
    @pytest.mark.parametrize("h0", [0.0, 0.25])
    def test_enumerated_rademacher_grid_equals_refitting_every_draw(self, h0):
        df = _wild_frame()
        W = np.array(list(itertools.product((1.0, -1.0), repeat=7)))
        beta, se, t_obs, t_star = _wcr_by_refitting(df, h0, W)
        res = sp.wild_cluster_bootstrap(
            df, "y", ["x", "z"], "cl", test_var="x", h0=h0, n_boot=999
        )
        assert res["enumerated"] and res["n_boot"] == 128
        assert res["beta_hat"] == pytest.approx(beta, rel=1e-12)
        assert res["se_cluster"] == pytest.approx(se, rel=1e-10)
        assert res["t_stat"] == pytest.approx(t_obs, rel=1e-10)
        assert res["p_boot"] == pytest.approx(_strict_p(t_star, t_obs, W), abs=1e-12)
        assert res["p_cluster"] == pytest.approx(
            2 * stats.t.sf(abs(t_obs), 6), rel=1e-10
        )
        lo, hi = np.percentile(t_star, [2.5, 97.5])
        assert res["ci_boot"][0] == pytest.approx(beta - hi * se, abs=1e-8)
        assert res["ci_boot"][1] == pytest.approx(beta - lo * se, abs=1e-8)

    @pytest.mark.parametrize("weight_type", ["rademacher", "webb", "mammen"])
    def test_sampled_weights_are_the_seeded_generator(self, weight_type):
        df = _wild_frame()
        rng = np.random.default_rng(42)
        if weight_type == "rademacher":
            draw = lambda: rng.choice([-1.0, 1.0], size=7)  # noqa: E731
        elif weight_type == "webb":
            vals = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5)])
            vals = np.r_[vals, -vals[::-1]]
            draw = lambda: rng.choice(vals, size=7)  # noqa: E731
        else:
            p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
            vals = np.array([-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2])
            draw = lambda: rng.choice(vals, size=7, p=[p, 1 - p])  # noqa: E731
        W = np.array([draw() for _ in range(99)])
        _, _, t_obs, t_star = _wcr_by_refitting(df, 0.0, W)
        res = sp.wild_cluster_bootstrap(
            df,
            "y",
            ["x", "z"],
            "cl",
            test_var="x",
            n_boot=99,
            weight_type=weight_type,
            seed=42,
        )
        assert not res["enumerated"]
        assert res["p_boot"] == pytest.approx(_strict_p(t_star, t_obs, W), abs=1e-12)

    def test_result_based_entry_point_agrees_and_reports_the_draws(self):
        df = _wild_frame()
        W = np.array(list(itertools.product((1.0, -1.0), repeat=7)))
        _, _, t_obs, t_star = _wcr_by_refitting(df, 0.1, W)
        fit = sp.regress("y ~ x + z", data=df, cluster="cl")
        res = sp.wild_cluster_boot(fit, df, "cl", "x", h0=0.1, n_boot=999)
        np.testing.assert_allclose(res["t_distribution"], t_star, rtol=1e-7, atol=1e-9)
        assert res["p_boot"] == pytest.approx(_strict_p(t_star, t_obs, W), abs=1e-12)
        assert res["h0"] == 0.1

    def test_inverted_interval_is_dual_to_the_test(self):
        df = _wild_frame()
        fit = sp.regress("y ~ x + z", data=df, cluster="cl")
        res = sp.wild_cluster_boot(
            fit, df, "cl", "x", n_boot=999, alpha=0.1, confidence_set=True
        )
        lo, hi = res["ci_inverted"]
        assert lo < res["beta_hat"] < hi
        step = 1e-6 * res["se_cluster"]

        def p_at(null):
            return sp.wild_cluster_boot(fit, df, "cl", "x", h0=null, n_boot=999)[
                "p_boot"
            ]

        # enumerated grid: the p-value is an exact step function of the null
        assert p_at(lo + step) >= 0.1 and p_at(hi - step) >= 0.1
        assert p_at(lo - step) < 0.1 and p_at(hi + step) < 0.1

    def test_subcluster_draws_flip_subclusters_and_cluster_the_variance(self):
        df = _wild_frame(n_clusters=4, size=10)
        W = np.array(list(itertools.product((1.0, -1.0), repeat=8)))
        beta, se, t_obs, t_star = _wcr_by_refitting(df, 0.0, W, boot_col="sub")
        res = sp.subcluster_wild_bootstrap(
            df,
            "y",
            ["x", "z"],
            "cl",
            subcluster="sub",
            test_var="x",
            n_boot=999,
            weight_type="rademacher",
        )
        assert res["n_clusters"] == 4 and res["n_subclusters"] == 8
        assert res["enumerated"] and res["n_boot"] == 256
        assert res["se_cluster"] == pytest.approx(se, rel=1e-10)
        assert res["p_boot"] == pytest.approx(_strict_p(t_star, t_obs, W), abs=1e-12)

    def test_subcluster_equal_to_cluster_is_the_plain_wild_bootstrap(self):
        df = _wild_frame()
        a = sp.subcluster_wild_bootstrap(
            df, "y", ["x", "z"], "cl", subcluster="cl", test_var="x", seed=3, n_boot=99
        )
        b = sp.wild_cluster_bootstrap(
            df, "y", ["x", "z"], "cl", test_var="x", seed=3, n_boot=99,
            weight_type="webb",
        )  # fmt: skip
        assert a["p_boot"] == b["p_boot"]
        assert a["ci_boot"] == pytest.approx(b["ci_boot"], rel=1e-12)

    def test_missing_rows_and_default_test_variable(self):
        df = _wild_frame()
        holes = df.assign(z=df["z"].where(df.index >= 6))
        a = sp.wild_cluster_bootstrap(holes, "y", ["z", "x"], "cl", n_boot=999)
        b = sp.wild_cluster_bootstrap(
            holes.dropna().reset_index(drop=True),
            "y",
            ["z", "x"],
            "cl",
            test_var="x",
            n_boot=999,
        )
        assert a["n_obs"] == len(df) - 6
        assert a["p_boot"] == b["p_boot"]
        assert a["beta_hat"] == pytest.approx(b["beta_hat"], rel=1e-12)

    def test_invalid_inputs(self):
        df = _wild_frame()
        with pytest.raises(ValueError, match="must be in x"):
            sp.wild_cluster_bootstrap(df, "y", ["x"], "cl", test_var="z")
        with pytest.raises(ValueError, match="at least two clusters"):
            sp.wild_cluster_bootstrap(df.assign(cl=1), "y", ["x"], "cl")
        with pytest.raises(ValueError, match="weight_type"):
            sp.wild_cluster_bootstrap(df, "y", ["x"], "cl", weight_type="normal")
        with pytest.raises(ValueError, match="at least two clusters"):
            sp.subcluster_wild_bootstrap(df.assign(cl=1), "y", ["x"], "cl")
        with pytest.warns(UserWarning, match="Only 3 clusters"):
            sp.wild_cluster_bootstrap(df[df.cl < 3], "y", ["x"], "cl", n_boot=99)

    def test_result_based_entry_point_refuses_a_single_cluster(self):
        df = _wild_frame()
        fit = sp.regress("y ~ x + z", data=df, cluster="cl")
        with pytest.raises(ValueError, match="(?i)cluster"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sp.wild_cluster_boot(fit, df.assign(cl=0), "cl", "x", n_boot=99)


class TestIvWildBootstrap:
    def _data(self):
        rng = np.random.default_rng(5)
        G, m = 8, 15
        cl = np.repeat(np.arange(G), m)
        n = cl.size
        z1 = rng.normal(size=n)
        w = rng.normal(size=n)
        u = rng.normal(size=n) + rng.normal(size=G)[cl]
        d = 0.8 * z1 + 0.3 * w + 0.6 * u + rng.normal(size=n)
        y = 1 + 0.5 * d + 0.4 * w + u
        return pd.DataFrame({"y": y, "d": d, "z1": z1, "w": w, "cl": cl})

    @pytest.mark.parametrize("efficient", [True, False])
    def test_enumerated_draws_equal_refitting_two_stage_least_squares(self, efficient):
        df = self._data()
        n = len(df)
        y = df["y"].to_numpy()
        d = df["d"].to_numpy()
        cl = df["cl"].to_numpy()
        Xo = np.column_stack([np.ones(n), df["w"]])  # exogenous regressors
        Z = np.column_stack([Xo, df["z1"]])  # instruments
        PZ = Z @ np.linalg.solve(Z.T @ Z, Z.T)
        G = 8
        corr = (G / (G - 1)) * ((n - 1) / (n - 3))

        def tsls_t(yv, dv, b0):
            X = np.column_stack([Xo, dv])
            AX = PZ @ X
            bread = np.linalg.inv(X.T @ AX)
            b = bread @ AX.T @ yv
            e = yv - X @ b
            sc = np.zeros((G, 3))
            np.add.at(sc, cl, AX * e[:, None])
            V = corr * bread @ sc.T @ sc @ bread
            return b[2], np.sqrt(V[2, 2]), (b[2] - b0) / np.sqrt(V[2, 2])

        b0 = 0.2
        beta, se, t_obs = tsls_t(y, d, b0)
        # restricted structural residual and reduced form
        g = np.linalg.lstsq(Xo, y - b0 * d, rcond=None)[0]
        u_r = y - b0 * d - Xo @ g
        if efficient:
            coef = np.linalg.lstsq(np.column_stack([Z, u_r]), d, rcond=None)[0]
            d_base = Z @ coef[:3]
        else:
            d_base = PZ @ d
        v_r = d - d_base
        W = np.array(list(itertools.product((1.0, -1.0), repeat=G)))
        t_star = []
        for wg in W:
            eps = wg[cl]
            d_s = d_base + eps * v_r
            y_s = b0 * d_s + Xo @ g + eps * u_r
            t_star.append(tsls_t(y_s, d_s, b0)[2])
        t_star = np.array(t_star)

        fit = sp.ivreg("y ~ w + (d ~ z1)", data=df, cluster="cl")
        res = iv_wild_bootstrap(
            fit, df, "cl", "d", n_boot=999, beta0=b0, efficient=efficient
        )
        assert res["enumerated"] and res["n_boot"] == 256
        assert res["beta_hat"] == pytest.approx(beta, rel=1e-9)
        assert res["se_cluster"] == pytest.approx(se, rel=1e-8)
        np.testing.assert_allclose(res["t_distribution"], t_star, rtol=1e-6, atol=1e-8)
        expected_p = np.mean(np.abs(t_star) > abs(t_obs) * (1 + 1e-10))
        assert res["p_boot"] == pytest.approx(expected_p, abs=1e-12)
        assert res["method"] == ("WRE" if efficient else "WUR")

    def test_invalid_requests(self):
        df = self._data()
        fit = sp.ivreg("y ~ w + (d ~ z1)", data=df, cluster="cl")
        with pytest.raises(MethodIncompatibility, match="endogenous"):
            iv_wild_bootstrap(fit, df, "cl", "w", n_boot=99)
        with pytest.raises(ValueError, match="not found"):
            iv_wild_bootstrap(fit, df, "cl", "nope", n_boot=99)
        with pytest.raises(MethodIncompatibility, match="row-aligned"):
            iv_wild_bootstrap(fit, df.iloc[:-1], "cl", "d", n_boot=99)
        ols = sp.regress("y ~ w + d", data=df)
        with pytest.raises(MethodIncompatibility, match="2SLS"):
            iv_wild_bootstrap(ols, df, "cl", "d", n_boot=99)


# --------------------------------------------------------------------- #
#  behaviour added with the fixes of this scan
# --------------------------------------------------------------------- #


class TestRefusalsAddedWithTheFixes:
    def test_bootstrap_refuses_a_single_block(self):
        df = _clustered().assign(g=1)
        with pytest.raises(DataInsufficient, match="block"):
            sp.bootstrap(df, _mean, n_boot=50, block="g", seed=1)

    def test_jackknife_se_refuses_absorbed_fixed_effects(self):
        df = _reg_frame()
        fit = sp.feols("y ~ x1 + x2 | g", data=df)
        with pytest.raises(MethodIncompatibility, match="fixed effects"):
            sp.jackknife_se(fit, df, "g")

    def test_jackknife_se_refuses_a_design_that_is_not_the_fit(self):
        # the cluster key cannot be aligned to the fitted rows, the formula
        # re-parse drops the interaction: no refit of this model exists
        df = _reg_frame()
        fit = sp.regress("y ~ x1 * x2", data=df, cluster="g")
        shorter = df.iloc[:150]
        with pytest.raises(MethodIncompatibility, match="does not reproduce"):
            sp.jackknife_se(fit, shorter, "g")

    def test_jackknife_se_with_factor_terms(self):
        df = _reg_frame().assign(k=lambda d: d.index % 3)
        X = np.column_stack(
            [
                np.ones(len(df)),
                (df["k"] == 1).astype(float),
                (df["k"] == 2).astype(float),
                df["x1"],
            ]
        )
        fit = sp.regress("y ~ x1 + C(k)", data=df, cluster="g")
        jk = sp.jackknife_se(fit, df, "g")
        expected = _cluster_jackknife(X, df["y"].to_numpy(), df["g"].to_numpy())
        got = jk.std_errors[["Intercept", "C(k)[T.1]", "C(k)[T.2]", "x1"]].to_numpy()
        np.testing.assert_allclose(got, expected, rtol=1e-8)
