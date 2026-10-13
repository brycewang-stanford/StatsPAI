"""Randomization inference: the permutation scheme and the p-value.

``sp.fisher_exact`` and ``sp.ri_test`` are compared with full enumerations
written here (tiny designs) and with loops that use the same generator.
The p-value convention checked throughout: the share of assignments whose
statistic is at least as extreme as the observed one, the observed
assignment being one of the enumerated rows; under sampling nothing is
added to numerator or denominator.
"""

import warnings
from itertools import combinations, product

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _diff(y, d):
    return y[d == 1].mean() - y[d == 0].mean()


def _all_assignments(n, k):
    out = np.zeros((0, n))
    rows = []
    for c in combinations(range(n), k):
        a = np.zeros(n)
        a[list(c)] = 1.0
        rows.append(a)
    return np.array(rows) if rows else out


def _nested_design():
    """Eight clusters of five in two strata, two treated clusters in each."""
    rng = np.random.default_rng(3)
    cl = np.repeat(np.arange(8), 5)
    st = np.repeat([0, 0, 0, 0, 1, 1, 1, 1], 5)
    d = np.array([1, 1, 0, 0, 1, 1, 0, 0])[cl].astype(float)
    y = rng.normal(size=40) + 0.5 * d + 2.0 * st
    return pd.DataFrame({"y": y, "d": d, "cl": cl, "st": st})


def _stratified_cluster_p(df):
    y, cl = df["y"].to_numpy(), df["cl"].to_numpy()
    obs = _diff(y, df["d"].to_numpy())
    halves = [list(combinations(block, 2)) for block in ([0, 1, 2, 3], [4, 5, 6, 7])]
    stats_ = []
    for a, b in product(*halves):
        treated = np.isin(cl, list(a) + list(b)).astype(float)
        stats_.append(_diff(y, treated))
    stats_ = np.array(stats_)
    return len(stats_), float(np.mean(np.abs(stats_) >= abs(obs) - 1e-12))


class TestEnumeration:
    def test_complete_randomization_counts_every_assignment_once(self):
        rng = np.random.default_rng(1)
        d = np.r_[np.ones(3), np.zeros(6)]
        y = rng.normal(size=9) + d
        df = pd.DataFrame({"y": y, "d": d})
        A = _all_assignments(9, 3)
        perm = np.array([_diff(y, a) for a in A])
        obs = _diff(y, d)
        two = np.mean(np.abs(perm) >= abs(obs) - 1e-12)
        one = np.mean(perm >= obs - 1e-12)
        fe = sp.fisher_exact(df, "y", "d")
        ri = sp.ri_test(df, "y", "d")
        assert fe.n_perm == ri["n_perms"] == 84
        assert ri["exact"]
        assert fe.p_value == pytest.approx(two, abs=1e-12)
        assert fe.p_one_sided == pytest.approx(one, abs=1e-12)
        assert ri["p_value"] == pytest.approx(two, abs=1e-12)
        assert ri["p_one_sided"] == pytest.approx(one, abs=1e-12)
        # the observed assignment is one of the rows, so p >= 1 / 84
        assert fe.p_value >= 1 / 84 - 1e-12
        np.testing.assert_allclose(np.sort(fe.perm_dist), np.sort(perm), atol=1e-12)

    @pytest.mark.parametrize(
        "fe_stat, ri_stat", [("ate", "diff_means"), ("t", "t"), ("ks", "ks")]
    )
    def test_the_two_entry_points_agree_on_an_enumerated_design(self, fe_stat, ri_stat):
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"y": rng.normal(size=9), "d": [1, 1, 1, 0, 0, 0, 0, 0, 0.0]})
        fe = sp.fisher_exact(df, "y", "d", statistic=fe_stat)
        ri = sp.ri_test(df, "y", "d", stat=ri_stat)
        assert fe.p_value == pytest.approx(ri["p_value"], abs=1e-12)
        assert fe.p_one_sided == pytest.approx(ri["p_one_sided"], abs=1e-12)
        assert fe.statistic == pytest.approx(ri["observed"], abs=1e-12)

    def test_stratified_enumeration_keeps_each_stratums_treated_count(self):
        rng = np.random.default_rng(4)
        st = np.repeat([0, 1, 2], 4)
        d = np.tile([1, 0, 0, 0], 3).astype(float)
        d[4:8] = [1, 1, 0, 0]
        y = rng.normal(size=12) + st + d
        df = pd.DataFrame({"y": y, "d": d, "st": st})
        blocks = [
            _all_assignments(4, 1),
            _all_assignments(4, 2),
            _all_assignments(4, 1),
        ]
        perm = np.array([_diff(y, np.concatenate(c)) for c in product(*blocks)])
        expected = np.mean(np.abs(perm) >= abs(_diff(y, d)) - 1e-12)
        fe = sp.fisher_exact(df, "y", "d", stratify="st")
        ri = sp.ri_test(df, "y", "d", strata="st")
        assert fe.n_perm == ri["n_perms"] == 4 * 6 * 4
        assert fe.p_value == pytest.approx(expected, abs=1e-12)
        assert ri["p_value"] == pytest.approx(expected, abs=1e-12)

    def test_cluster_enumeration_moves_whole_clusters(self):
        df = _nested_design()
        y, cl = df["y"].to_numpy(), df["cl"].to_numpy()
        perm = np.array(
            [_diff(y, np.isin(cl, c).astype(float)) for c in combinations(range(8), 4)]
        )
        expected = np.mean(np.abs(perm) >= abs(_diff(y, df["d"].to_numpy())) - 1e-12)
        fe = sp.fisher_exact(df, "y", "d", cluster="cl")
        ri = sp.ri_test(df, "y", "d", cluster="cl")
        assert fe.n_perm == ri["n_perms"] == 70
        assert fe.p_value == pytest.approx(expected, abs=1e-12)
        assert ri["p_value"] == pytest.approx(expected, abs=1e-12)

    def test_ri_test_permutes_clusters_within_strata(self):
        df = _nested_design()
        count, expected = _stratified_cluster_p(df)
        ri = sp.ri_test(df, "y", "d", cluster="cl", strata="st")
        assert ri["n_perms"] == count == 36
        assert ri["p_value"] == pytest.approx(expected, abs=1e-12)

    def test_fisher_exact_permutes_clusters_within_strata(self):
        df = _nested_design()
        count, expected = _stratified_cluster_p(df)
        fe = sp.fisher_exact(df, "y", "d", cluster="cl", stratify="st")
        assert fe.n_perm == count
        assert fe.p_value == pytest.approx(expected, abs=1e-12)


class TestSampledPermutations:
    def test_unrestricted_draws_are_the_seeded_generator(self):
        rng = np.random.default_rng(0)
        d = (np.arange(40) % 2).astype(float)
        y = rng.normal(size=40) + 0.4 * d
        df = pd.DataFrame({"y": y, "d": d})
        gen = np.random.default_rng(17)
        perm = np.array([_diff(y, gen.permutation(d)) for _ in range(300)])
        fe = sp.fisher_exact(df, "y", "d", n_perm=300, seed=17)
        np.testing.assert_allclose(fe.perm_dist, perm, atol=1e-12)
        # no +1 in numerator or denominator
        expected = np.mean(np.abs(perm) >= abs(_diff(y, d)) - 1e-12)
        assert fe.p_value == pytest.approx(expected, abs=1e-12)
        assert fe.n_perm == 300

    def test_stratified_draws_permute_within_strata(self):
        rng = np.random.default_rng(0)
        st = np.repeat(np.arange(4), 10)
        d = np.tile([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 4).astype(float)
        y = rng.normal(size=40) + st
        df = pd.DataFrame({"y": y, "d": d, "st": st})
        fe = sp.fisher_exact(df, "y", "d", stratify="st", n_perm=200, seed=3)
        gen = np.random.default_rng(3)
        perm = []
        for _ in range(200):
            dp = np.zeros(40)
            for s in range(4):
                idx = np.flatnonzero(st == s)
                dp[idx] = gen.permutation(d[idx])
            perm.append(_diff(y, dp))
        np.testing.assert_allclose(fe.perm_dist, perm, atol=1e-12)
        ri = sp.ri_test(df, "y", "d", strata="st", n_perms=200, seed=3)
        np.testing.assert_allclose(ri["perm_distribution"], perm, atol=1e-12)
        assert not ri["exact"]

    def test_cluster_draws_assign_one_value_per_cluster(self):
        rng = np.random.default_rng(0)
        cl = np.repeat(np.arange(16), 3)
        dc = (np.arange(16) % 2).astype(float)
        y = rng.normal(size=48) + rng.normal(size=16)[cl]
        df = pd.DataFrame({"y": y, "d": dc[cl], "cl": cl})
        gen = np.random.default_rng(9)
        perm = np.array([_diff(y, gen.permutation(dc)[cl]) for _ in range(150)])
        fe = sp.fisher_exact(df, "y", "d", cluster="cl", n_perm=150, seed=9)
        np.testing.assert_allclose(fe.perm_dist, perm, atol=1e-12)


class TestStatistics:
    def _frame(self):
        rng = np.random.default_rng(5)
        n = 60
        st = np.arange(n) % 3
        x = rng.normal(size=n)
        d = (rng.permutation(n) < 25).astype(float)
        y = 1 + 0.5 * d + x + 0.3 * st + rng.normal(size=n)
        return pd.DataFrame({"y": np.round(y, 1), "d": d, "x": x, "st": st})

    def test_rank_sum_is_centred_at_its_null_mean(self):
        df = self._frame()
        ranks = stats.rankdata(df["y"])  # mid-ranks; y is rounded, so ties
        n1 = int(df["d"].sum())
        expected = ranks[df["d"] == 1].sum() - n1 * (len(df) + 1) / 2
        ri = sp.ri_test(df, "y", "d", stat="rank_sum", n_perms=50, seed=0)
        assert ri["observed"] == pytest.approx(expected, abs=1e-10)

    def test_t_is_the_unequal_variance_statistic(self):
        df = self._frame()
        a, b = df.loc[df.d == 1, "y"], df.loc[df.d == 0, "y"]
        expected = stats.ttest_ind(a, b, equal_var=False).statistic
        ri = sp.ri_test(df, "y", "d", stat="t", n_perms=50, seed=0)
        fe = sp.fisher_exact(df, "y", "d", statistic="t", n_perm=50, seed=0)
        assert ri["observed"] == pytest.approx(expected, rel=1e-10)
        assert fe.statistic == pytest.approx(expected, rel=1e-10)

    def test_ols_statistic_is_the_regression_with_stratum_effects(self):
        df = self._frame()
        X = np.column_stack(
            [np.ones(len(df)), df["d"], df["x"], df["st"] == 1, df["st"] == 2]
        ).astype(float)
        fit = sm.OLS(df["y"].to_numpy(), X).fit(cov_type="HC1")
        coef = sp.ri_test(
            df, "y", "d", stat="ols", covariates=["x"], strata="st", n_perms=30, seed=0
        )
        tval = sp.ri_test(
            df,
            "y",
            "d",
            stat="ols_t",
            covariates=["x"],
            strata="st",
            n_perms=30,
            seed=0,
        )
        assert coef["observed"] == pytest.approx(fit.params[1], abs=1e-10)
        assert tval["observed"] == pytest.approx(fit.tvalues[1], rel=1e-8)

    def test_lin_statistic_is_the_interacted_regression_with_hc2(self):
        df = self._frame()
        xc = df["x"] - df["x"].mean()
        X = np.column_stack([np.ones(len(df)), df["d"], xc, df["d"] * xc])
        fit = sm.OLS(df["y"].to_numpy(), X).fit(cov_type="HC2")
        coef = sp.ri_test(df, "y", "d", stat="lin", covariates=["x"], n_perms=30)
        tval = sp.ri_test(df, "y", "d", stat="lin_t", covariates=["x"], n_perms=30)
        assert coef["observed"] == pytest.approx(fit.params[1], abs=1e-10)
        assert tval["observed"] == pytest.approx(fit.tvalues[1], rel=1e-8)

    def test_callable_statistic_and_controls(self):
        df = self._frame()
        ri = sp.ri_test(
            df, "y", "d", stat=lambda y, d: float(np.median(y[d == 1])), n_perms=20
        )
        assert ri["observed"] == pytest.approx(df.loc[df.d == 1, "y"].median())
        # fisher_exact(controls=) tests on the residuals of y on the controls
        X = sm.add_constant(df[["x"]].to_numpy())
        resid = df["y"].to_numpy() - X @ np.linalg.lstsq(X, df["y"], rcond=None)[0]
        fe = sp.fisher_exact(df, "y", "d", controls=["x"], n_perm=50, seed=0)
        assert fe.statistic == pytest.approx(
            _diff(resid, df["d"].to_numpy()), abs=1e-10
        )


class TestRefusals:
    def test_ri_test_refuses_what_it_cannot_permute(self):
        df = _nested_design()
        flipped = df.assign(d=df["d"].where(df.index % 5 != 0, 1 - df["d"]))
        with pytest.raises(MethodIncompatibility, match="varies within"):
            sp.ri_test(flipped, "y", "d", cluster="cl")
        with pytest.raises(MethodIncompatibility, match="binary"):
            sp.ri_test(df.assign(d=np.arange(40) % 3), "y", "d")
        with pytest.raises(MethodIncompatibility, match="covariates"):
            sp.ri_test(df, "y", "d", covariates=["st"])
        with pytest.raises(MethodIncompatibility, match="needs covariates"):
            sp.ri_test(df, "y", "d", stat="lin")
        with pytest.raises(MethodIncompatibility, match="cluster"):
            sp.ri_test(df, "y", "d", stat="lin_t", covariates=["st"], cluster="cl")
        with pytest.raises(ValueError, match="Unknown stat"):
            sp.ri_test(df, "y", "d", stat="wald")
        with pytest.raises(ValueError, match="Unknown statistic"):
            sp.fisher_exact(df, "y", "d", statistic="wald")

    def test_ri_test_reports_rows_dropped_for_missing_values(self):
        df = _nested_design()
        holes = df.assign(y=df["y"].where(df.index >= 5))  # one whole cluster
        with pytest.warns(UserWarning, match="dropped 5 of 40"):
            ri = sp.ri_test(holes, "y", "d", cluster="cl")
        assert ri["n_perms"] == 35  # C(7, 3): cluster 0 was treated

    def test_fisher_exact_refuses_treatment_varying_within_clusters(self):
        df = _nested_design()
        flipped = df.assign(d=df["d"].where(df.index % 5 != 0, 1 - df["d"]))
        with pytest.raises(ValueError):
            sp.fisher_exact(flipped, "y", "d", cluster="cl", n_perm=200, seed=0)

    def test_fisher_exact_refuses_a_non_binary_treatment(self):
        df = _nested_design().assign(d=np.arange(40) % 3.0)
        with pytest.raises(ValueError):
            sp.fisher_exact(df, "y", "d", n_perm=100, seed=0)

    def test_fisher_exact_refuses_a_single_arm(self):
        df = _nested_design().assign(d=1.0)
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sp.fisher_exact(df, "y", "d", n_perm=100, seed=0)

    def test_fisher_exact_refuses_alpha_outside_the_unit_interval(self):
        with pytest.raises(ValueError):
            sp.fisher_exact(_nested_design(), "y", "d", n_perm=100, seed=0, alpha=2)


class TestStratifiedClusterSampling:
    def _support(self, df):
        y, cl = df["y"].to_numpy(), df["cl"].to_numpy()
        halves = [list(combinations(b, 2)) for b in ([0, 1, 2, 3], [4, 5, 6, 7])]
        return np.array(
            [
                _diff(y, np.isin(cl, list(a) + list(b)).astype(float))
                for a, b in product(*halves)
            ]
        )

    def test_fisher_exact_samples_clusters_within_strata(self):
        # 36 assignments, n_perm=25: the draws are sampled, and each must be
        # one of the 36 the stratified design allows (34 of the 70
        # unstratified cluster assignments are not).
        df = _nested_design()
        support = self._support(df)
        fe = sp.fisher_exact(
            df, "y", "d", cluster="cl", stratify="st", n_perm=25, seed=4
        )
        assert fe.n_perm == 25
        gaps = np.abs(fe.perm_dist[:, None] - support[None, :]).min(axis=1)
        assert gaps.max() < 1e-12

    def test_fisher_exact_refuses_strata_that_split_a_cluster(self):
        df = _nested_design()
        split = df.assign(st=np.arange(len(df)) % 2)
        with pytest.raises(MethodIncompatibility, match="varies within"):
            sp.fisher_exact(split, "y", "d", cluster="cl", stratify="st")
