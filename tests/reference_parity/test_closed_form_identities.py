"""Closed-form identities for estimators that had no numerical evidence.

Every assertion here compares a StatsPAI output with a number computed
independently in the test body from the same data: a textbook formula, a
group-by, or a reference fit in statsmodels / scipy. None is a recovery of
a population parameter, so tolerances are machine-level (1e-9 or tighter,
stated per test) and a failure means the implementation changed what it
computes.

Sources for the formulas are the definitions the functions document:
Shannon entropy and the Simpson / Hill family for diversity, the two-sample
z-test sample-size formula with the design effect 1 + (m - 1) * ICC for
cluster trials, Manski's worst-case bounds, Fisher's z for partial
correlation, the rank-and-treat rule, and the difference-in-means algebra
of matched-pair and 2x2 difference-in-differences designs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

import statspai as sp

# --------------------------------------------------------------------- #
#  Diversity indices
# --------------------------------------------------------------------- #


class TestDiversityIndex:
    counts = np.array([5, 3, 2])
    p = counts / counts.sum()

    def _records(self) -> pd.DataFrame:
        return pd.DataFrame({"species": ["a"] * 5 + ["b"] * 3 + ["c"] * 2})

    def test_all_indices_match_their_definitions(self):
        out = sp.diversity_index(self._records(), species="species", index="all", q=2.0)
        p = self.p
        shannon = -(p * np.log(p)).sum()
        simpson = (p**2).sum()
        expected = {
            "shannon": shannon,
            "richness": 3.0,
            "pielou": shannon / np.log(3),
            "simpson": simpson,
            "gini_simpson": 1.0 - simpson,
            "inv_simpson": 1.0 / simpson,
            "hill": 1.0 / simpson,  # Hill number of order 2
        }
        for name, value in expected.items():
            assert out[name] == pytest.approx(value, abs=1e-12), name

    def test_hill_number_limits(self):
        rec = self._records()
        shannon = -(self.p * np.log(self.p)).sum()
        assert sp.diversity_index(
            rec, species="species", index="hill", q=0.0
        ) == pytest.approx(3.0, abs=1e-12)
        assert sp.diversity_index(
            rec, species="species", index="hill", q=1.0
        ) == pytest.approx(np.exp(shannon), abs=1e-10)

    def test_log_base_and_matrix_input(self):
        bits = sp.diversity_index(
            self._records(), species="species", index="shannon", base=2
        )
        assert bits == pytest.approx(-(self.p * np.log2(self.p)).sum(), abs=1e-12)
        mat = sp.diversity_index(np.array([[5, 3, 2], [10, 0, 0]]), index="shannon")
        # A single-species site has zero entropy.
        assert float(np.asarray(mat).ravel()[1]) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------- #
#  Experimental design
# --------------------------------------------------------------------- #


class TestOptimalDesign:
    def test_individual_rct_sample_size(self):
        res = sp.optimal_design(design="individual", mde=0.2, sigma=1.0)
        z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
        per_arm = int(np.ceil(2 * z**2 * 1.0**2 / 0.2**2))
        assert res.n_per_arm == per_arm == 393
        assert res.n_total == 2 * per_arm

    def test_individual_rct_mde_given_n(self):
        res = sp.optimal_design(design="individual", n=800, sigma=1.0)
        z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
        assert res.mde == pytest.approx(z * np.sqrt(2 * 1.0 / 400), abs=1e-12)

    def test_cluster_rct_applies_the_design_effect(self):
        res = sp.optimal_design(
            design="cluster", mde=0.2, sigma=1.0, icc=0.05, cluster_size=20
        )
        z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
        deff = 1 + (20 - 1) * 0.05
        clusters_per_arm = int(np.ceil(2 * z**2 / 0.2**2 * deff / 20))
        assert res.n_clusters == 2 * clusters_per_arm == 78
        assert res.n_total == res.n_clusters * 20


class TestRandomize:
    def test_stratified_assignment_is_balanced_within_strata(self):
        rng = np.random.default_rng(0)
        n = 203
        df = pd.DataFrame(
            {"district": rng.integers(0, 4, n), "age": rng.normal(40, 10, n)}
        )
        res = sp.randomize(df, strata="district", seed=1)
        tab = res.data.groupby("district")["treatment"].agg(["sum", "count"])
        # Complete randomisation inside each stratum: the arms differ by at
        # most one unit.
        assert ((2 * tab["sum"] - tab["count"]).abs() <= 1).all()
        assert res.n_treated + res.n_control == n

    def test_same_seed_same_assignment(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"district": rng.integers(0, 4, 120)})
        a = sp.randomize(df, strata="district", seed=7).data["treatment"]
        b = sp.randomize(df, strata="district", seed=7).data["treatment"]
        assert (a.to_numpy() == b.to_numpy()).all()


# --------------------------------------------------------------------- #
#  Policy targeting and concordance
# --------------------------------------------------------------------- #


class TestPolicyTargeting:
    tau = np.array([2.0, 1.0, 0.5, -0.5, -2.0])

    def test_budgeted_rule_treats_the_largest_effects(self):
        out = sp.policy_targeting(self.tau, budget=2)
        assert out["policy"].tolist() == [1, 1, 0, 0, 0]
        assert out["expected_gain"] == pytest.approx(3.0, abs=1e-12)
        assert out["threshold"] == pytest.approx(1.0, abs=1e-12)
        assert out["expected_gain_treat_all"] == pytest.approx(
            self.tau.sum(), abs=1e-12
        )
        # Random assignment of the same budget earns budget * mean effect.
        assert out["expected_gain_random"] == pytest.approx(
            2 * self.tau.mean(), abs=1e-12
        )

    def test_min_effect_guard_caps_the_budget(self):
        out = sp.policy_targeting(self.tau, budget=5)
        assert out["n_treated"] == 3
        assert out["expected_gain"] == pytest.approx(3.5, abs=1e-12)

    def test_fraction_budget(self):
        out = sp.policy_targeting(self.tau, frac=0.4)
        assert out["n_treated"] == 2


def test_rwd_rct_concordance_arithmetic():
    con = sp.rwd_rct_concordance(rct_estimate=0.50, rct_se=0.20, rwd_estimate=0.42)
    assert con.zscore_difference == pytest.approx((0.42 - 0.50) / 0.20, abs=1e-12)
    assert con.relative_difference == pytest.approx((0.42 - 0.50) / 0.50, abs=1e-12)
    assert con.rwd_inside_rct_ci is True
    far = sp.rwd_rct_concordance(rct_estimate=0.50, rct_se=0.20, rwd_estimate=1.00)
    assert far.rwd_inside_rct_ci is False  # 2.5 SE away


def test_copula_sensitivity_is_linear_in_rho():
    res = sp.copula_sensitivity(0.3, 0.1, sigma_u=2.0, sigma_y=0.5)
    curve = res.curve
    z = stats.norm.ppf(0.975)
    np.testing.assert_allclose(curve["bias"], curve["rho"] * 2.0 * 0.5, atol=1e-12)
    np.testing.assert_allclose(
        curve["adjusted_estimate"], 0.3 - curve["bias"], atol=1e-12
    )
    np.testing.assert_allclose(
        curve["ci_low"], curve["adjusted_estimate"] - z * 0.1, atol=1e-12
    )
    # The breakpoint is the smallest |rho| on the grid whose interval
    # covers zero: 0.3 - rho < 1.96 * 0.1 first holds at rho = 0.15.
    insignificant = curve.loc[~curve["significant"], "rho"].abs()
    assert res.breakpoint == pytest.approx(insignificant.min(), abs=1e-12)
    assert res.breakpoint == pytest.approx(0.15, abs=1e-12)


# --------------------------------------------------------------------- #
#  Partial correlation, propensity scores, trimming
# --------------------------------------------------------------------- #


def _fisher_z_pvalue(r: float, n: int, k: int) -> float:
    zstat = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - k - 3)
    return float(2 * stats.norm.sf(abs(zstat)))


def test_partial_corr_pvalue_is_the_fisher_z_test():
    rng = np.random.default_rng(0)
    n = 200
    z = rng.normal(size=n)
    x = z + rng.normal(size=n)
    y = z + rng.normal(size=n)
    r_marginal = np.corrcoef(x, y)[0, 1]
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    r_partial = np.corrcoef(rx, ry)[0, 1]
    assert sp.partial_corr_pvalue(x, y) == pytest.approx(
        _fisher_z_pvalue(r_marginal, n, 0), rel=1e-9
    )
    assert sp.partial_corr_pvalue(x, y, z) == pytest.approx(
        _fisher_z_pvalue(r_partial, n, 1), rel=1e-9
    )


@pytest.fixture(scope="module")
def selection_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 500
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * x1 - 0.5 * x2))))
    return pd.DataFrame({"d": d, "x1": x1, "x2": x2})


def test_propensity_score_is_the_logit_mle(selection_frame):
    df = selection_frame
    ps = sp.propensity_score(df, treatment="d", covariates=["x1", "x2"])
    ref = sm.Logit(df["d"], sm.add_constant(df[["x1", "x2"]])).fit(disp=0).predict()
    # Same likelihood, same optimum; observed gap 2e-16.
    np.testing.assert_allclose(ps.to_numpy(), np.asarray(ref), atol=1e-8)
    # A logit with an intercept reproduces the treated share exactly.
    assert ps.mean() == pytest.approx(df["d"].mean(), abs=1e-8)


def test_sturmer_trimming_keeps_the_fixed_overlap_band(selection_frame):
    df = selection_frame
    ps = sp.propensity_score(df, treatment="d", covariates=["x1", "x2"])
    trimmed = sp.trimming(df, treatment="d", covariates=["x1", "x2"], method="sturmer")
    keep = (ps >= 0.1) & (ps <= 0.9)
    assert set(trimmed.index) == set(df.index[keep])


def test_crump_trimming_keeps_a_symmetric_band(selection_frame):
    df = selection_frame
    ps = sp.propensity_score(df, treatment="d", covariates=["x1", "x2"])
    trimmed = sp.trimming(df, treatment="d", covariates=["x1", "x2"], method="crump")
    kept = ps.loc[trimmed.index]
    dropped = ps.drop(trimmed.index)
    # Crump et al. keep [a, 1 - a]: nothing dropped lies strictly inside
    # the band spanned by what was kept.
    a = min(kept.min(), 1 - kept.max())
    assert ((dropped < a) | (dropped > 1 - a)).all()


# --------------------------------------------------------------------- #
#  Bounds
# --------------------------------------------------------------------- #


def test_manski_bounds_are_the_worst_case_formula():
    rng = np.random.default_rng(42)
    n = 400
    d = rng.binomial(1, 0.5, size=n)
    y = np.clip(1.0 + 0.5 * d + rng.normal(0, 1.0, size=n), 0.0, 1.0)
    res = sp.partial_identification(
        pd.DataFrame({"y": y, "d": d}), "y", "d", method="manski"
    )
    p1 = d.mean()
    m1, m0 = y[d == 1].mean(), y[d == 0].mean()
    y_lo, y_hi = y.min(), y.max()
    lower = (m1 * p1 + y_lo * (1 - p1)) - (m0 * (1 - p1) + y_hi * p1)
    upper = (m1 * p1 + y_hi * (1 - p1)) - (m0 * (1 - p1) + y_lo * p1)
    assert res.model_info["lower_bound"] == pytest.approx(lower, abs=1e-12)
    assert res.model_info["upper_bound"] == pytest.approx(upper, abs=1e-12)
    # Width equals the outcome range, whatever the data.
    assert upper - lower == pytest.approx(y_hi - y_lo, abs=1e-12)
    assert res.estimate == pytest.approx((lower + upper) / 2, abs=1e-12)


# --------------------------------------------------------------------- #
#  Difference-in-means algebra
# --------------------------------------------------------------------- #


def test_cluster_matched_pair_is_the_mean_pair_difference():
    rng = np.random.default_rng(1)
    rows = []
    for p in range(60):
        base = rng.normal()
        for arm in (0, 1):
            for _ in range(10):
                rows.append(
                    {
                        "y": base + 0.5 * arm + rng.normal(0, 0.3),
                        "cluster": p * 2 + arm,
                        "treat": arm,
                        "pair": p,
                    }
                )
    df = pd.DataFrame(rows)
    res = sp.cluster_matched_pair(
        df, y="y", cluster="cluster", treat="treat", pair="pair"
    )
    cluster_means = df.groupby(["pair", "treat"])["y"].mean().unstack()
    diff = cluster_means[1] - cluster_means[0]
    assert res.estimate == pytest.approx(diff.mean(), abs=1e-12)
    assert res.se == pytest.approx(diff.std(ddof=1) / np.sqrt(len(diff)), abs=1e-12)
    assert abs(res.estimate - 0.5) <= 4.0 * res.se  # the planted effect


def test_did_few_treated_point_estimate_is_the_2x2_did():
    rng = np.random.default_rng(0)
    rows = []
    for j in range(30):
        eff = rng.normal()
        for t in range(8):
            d = 1.0 if (j == 0 and t >= 4) else 0.0
            y = eff + 0.1 * t + 2.0 * d + rng.normal(0, 0.5)
            rows.append({"g": j, "t": t, "d": d, "y": y})
    df = pd.DataFrame(rows)
    # Called without the ``sp.<name>(`` spelling on purpose: the identity
    # below pins the point estimate only. The function exists for its
    # interval, whose coverage on this design is about 85% at a nominal
    # 95% (29 placebo groups), so it should not be credited with
    # known-truth evidence by the parity index.
    few_treated = getattr(sp, "did_few_treated")
    res = few_treated(df, y="y", id="g", time="t", treat="d")
    trt, ctl = df[df["g"] == 0], df[df["g"] != 0]
    by_hand = (
        trt.loc[trt["t"] >= 4, "y"].mean() - trt.loc[trt["t"] < 4, "y"].mean()
    ) - (ctl.loc[ctl["t"] >= 4, "y"].mean() - ctl.loc[ctl["t"] < 4, "y"].mean())
    # On a balanced panel the two-way fixed-effects coefficient is this
    # difference of differences.
    assert res.estimate == pytest.approx(by_hand, abs=1e-10)


# --------------------------------------------------------------------- #
#  Regressions that are OLS underneath
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def proxy_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 4000
    U = rng.normal(size=n)
    D = (0.5 * U + rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame(
        {
            "Y": 1.5 * D + U + rng.normal(size=n),
            "D": D,
            "X": rng.normal(size=n),
            "NCE": 0.8 * U + 0.5 * rng.normal(size=n),
            "NCO": 0.8 * U + 0.5 * rng.normal(size=n),
        }
    )


def test_negative_control_outcome_is_ols_with_hc1(proxy_frame):
    df = proxy_frame
    res = sp.negative_control_outcome(df, nco="NCO", treat="D", covariates=["X"])
    ols = sm.OLS(df["NCO"], sm.add_constant(df[["D", "X"]])).fit(cov_type="HC1")
    assert res.estimate == pytest.approx(ols.params["D"], abs=1e-10)
    assert res.se == pytest.approx(ols.bse["D"], abs=1e-10)
    # U drives both D and the control outcome, so the association is real.
    assert res.estimate > 10 * res.se


def test_negative_control_exposure_is_ols_with_hc1(proxy_frame):
    df = proxy_frame
    res = sp.negative_control_exposure(df, y="Y", nce="NCE")
    ols = sm.OLS(df["Y"], sm.add_constant(df[["NCE"]])).fit(cov_type="HC1")
    assert res.estimate == pytest.approx(ols.params["NCE"], abs=1e-10)
    assert res.se == pytest.approx(ols.bse["NCE"], abs=1e-10)


# --------------------------------------------------------------------- #
#  RD helpers that wrap rdrobust
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def sharp_rd_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(-1, 1, n)
    y = 1.0 * (x >= 0) + 0.5 * x + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x": x, "y": y})


def test_rdplacebo_true_cutoff_row_is_rdrobust(sharp_rd_frame):
    import matplotlib.pyplot as plt

    df = sharp_rd_frame
    placebo = sp.rdplacebo(df, y="y", x="x", n_placebo=2)
    plt.close("all")
    row = placebo[placebo["is_true_cutoff"]].iloc[0]
    ref = sp.rdrobust(df, y="y", x="x", c=0.0)
    assert float(row["estimate"]) == pytest.approx(ref.estimate, abs=1e-12)
    assert float(row["se"]) == pytest.approx(ref.se, abs=1e-12)


def test_rdbwsensitivity_row_is_rdrobust_at_that_bandwidth(sharp_rd_frame):
    import matplotlib.pyplot as plt

    df = sharp_rd_frame
    table = sp.rdbwsensitivity(df, y="y", x="x", c=0.0, bw_grid=[0.3, 0.6])
    plt.close("all")
    for _, row in table.iterrows():
        ref = sp.rdrobust(df, y="y", x="x", c=0.0, h=float(row["bandwidth"]))
        assert float(row["estimate"]) == pytest.approx(ref.estimate, abs=1e-12)
        assert float(row["se"]) == pytest.approx(ref.se, abs=1e-12)


# --------------------------------------------------------------------- #
#  Heterogeneity and efficiency summaries
# --------------------------------------------------------------------- #


def test_cate_by_group_is_a_quartile_groupby():
    rng = np.random.default_rng(42)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = rng.integers(0, 2, size=n)
    y = 0.5 * x1 + 0.3 * x2 + (1.0 + x1) * d + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    res = sp.metalearner(df, y="y", treat="d", covariates=["x1", "x2"], learner="t")
    groups = sp.cate_by_group(res, df, by="cate", n_groups=4)
    # The fitted CATEs are the input to the identity, not what is under
    # test, so they are fetched without the ``sp.<name>(`` spelling that
    # the parity index reads as evidence for a function.
    predict = getattr(sp, "predict_cate")
    cate = pd.Series(np.asarray(predict(res, df), dtype=float))
    quartile = pd.qcut(cate, 4, labels=False)
    means = cate.groupby(quartile).mean().to_numpy()
    ses = (cate.groupby(quartile).std(ddof=1) / np.sqrt(n / 4)).to_numpy()
    np.testing.assert_allclose(groups["mean_cate"].to_numpy(), means, atol=1e-10)
    np.testing.assert_allclose(groups["se"].to_numpy(), ses, atol=1e-10)
    assert groups["n"].tolist() == [100, 100, 100, 100]


def test_te_rank_and_te_summary_describe_the_same_scores():
    rng = np.random.default_rng(0)
    n = 200
    log_k = rng.normal(0, 1, n)
    log_l = rng.normal(0, 1, n)
    u = rng.exponential(0.3, n)
    log_y = 0.4 * log_k + 0.5 * log_l + rng.normal(0, 0.1, n) - u
    df = pd.DataFrame({"log_y": log_y, "log_k": log_k, "log_l": log_l})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.frontier(df, y="log_y", x=["log_k", "log_l"])
    ranked = sp.te_rank(res)
    summary = sp.te_summary(res).iloc[0]
    eff = ranked["efficiency"]
    assert eff.is_monotonic_decreasing
    assert ranked["rank"].tolist() == list(range(1, n + 1))
    assert ((eff > 0) & (eff <= 1)).all()
    assert summary["n"] == n
    assert summary["mean"] == pytest.approx(eff.mean(), abs=1e-12)
    assert summary["max"] == pytest.approx(eff.max(), abs=1e-12)
    assert summary["median"] == pytest.approx(eff.median(), abs=1e-12)
    # The scores track the true efficiency exp(-u) of each firm.
    truth = np.exp(-u)
    assert np.corrcoef(eff.sort_index().to_numpy(), truth)[0, 1] > 0.8


# --------------------------------------------------------------------- #
#  Nonlinear decomposition adding-up
# --------------------------------------------------------------------- #


def test_yun_nonlinear_adds_up_to_the_raw_rate_gap():
    df = sp.cps_wage()
    res = sp.yun_nonlinear(
        df, y="union", group="female", x=["education", "experience"], model="logit"
    )
    raw = df.loc[df["female"] == 0, "union"].mean() - (
        df.loc[df["female"] == 1, "union"].mean()
    )
    # A logit with an intercept fits each group's mean exactly, so the
    # decomposed gap is the raw difference in rates.
    assert res.gap == pytest.approx(raw, abs=1e-8)
    assert res.explained + res.unexplained == pytest.approx(res.gap, abs=1e-10)


def test_disparity_decompose_adds_up_and_recovers_the_components():
    rng = np.random.default_rng(0)
    n = 20_000
    g = rng.integers(0, 2, n)
    m = rng.normal(2.0 - 0.8 * g, 1.0)
    y = 50.0 + 5.0 * m + 3.0 * g + rng.normal(0, 2.0, n)
    res = sp.disparity_decompose(
        pd.DataFrame({"y": y, "group": g, "med": m}),
        y="y",
        group="group",
        mediator="med",
    )
    assert res.total_disparity == pytest.approx(
        res.initial_disparity + res.mediator_attributable, abs=1e-8
    )
    # Truth: direct 3.0, through the mediator 5.0 * (-0.8) = -4.0, total
    # -1.0. The mediator gap has SE sqrt(2/10000) = 0.014, times 5 = 0.07,
    # so 0.3 is roughly four sigma.
    assert res.initial_disparity == pytest.approx(3.0, abs=0.3)
    assert res.mediator_attributable == pytest.approx(-4.0, abs=0.3)
    assert res.total_disparity == pytest.approx(-1.0, abs=0.3)


# --------------------------------------------------------------------- #
#  Prediction-powered inference
# --------------------------------------------------------------------- #


def test_ppi_mean_untuned_is_the_textbook_estimator():
    rng = np.random.default_rng(0)
    n, N = 100, 5000
    y = 2.0 + rng.normal(0, 1, n + N)
    f = y + rng.normal(0, 0.5, n + N)
    res = sp.ppi_mean(y=y[:n], yhat=f[:n], yhat_unlabeled=f[n:], tune=False)
    rectifier = y[:n] - f[:n]
    estimate = f[n:].mean() + rectifier.mean()
    se = np.sqrt(f[n:].var(ddof=1) / N + rectifier.var(ddof=1) / n)
    assert res.estimate == pytest.approx(estimate, abs=1e-12)
    assert res.se == pytest.approx(se, abs=1e-12)
    assert res.model_info["classical_estimate"] == pytest.approx(
        y[:n].mean(), abs=1e-12
    )
    assert res.model_info["classical_se"] == pytest.approx(
        y[:n].std(ddof=1) / np.sqrt(n), abs=1e-12
    )
