"""Third batch of known-truth anchors from the 2026-10 pass.

Fixes. ``sp.kernel_iv`` (the structural function under confounding),
``sp.bcf_ordinal`` (standard error of the dose-level effects) and
``sp.bcf_longitudinal`` (second-pass bias; periods with no treatment
variation).

Recoveries. One simulated design per estimator with a planted answer.
Every design is fixed by its seed, so the bands below are deterministic
checks, quoted next to the Monte Carlo spread they were sized against.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning

pytestmark = pytest.mark.filterwarnings("ignore")


# --------------------------------------------------------------------- #
#  kernel_iv: recovers h under confounding, where regression does not
# --------------------------------------------------------------------- #


def _confounded_iv(seed: int, n: int = 3000) -> pd.DataFrame:
    """y = sin(d) + u + noise, d = 0.8 z + 0.5 u + v. E[u | d] is not 0."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = 0.8 * z + 0.5 * u + rng.normal(size=n)
    y = np.sin(d) + u + 0.3 * rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "z": z})


class TestKernelIVStructuralFunction:
    grid = np.linspace(-1.5, 1.5, 13)

    def _fit(self, seed: int):
        df = _confounded_iv(seed)
        res = sp.kernel_iv(
            df, y="y", treat="d", instrument="z", grid=self.grid, n_boot=60, seed=seed
        )
        return df, res

    def test_recovers_the_structural_function(self):
        # Across 20 seeds the RMSE over this grid was 0.05 to 0.06. The
        # estimator this replaced sat at 0.25 on the same designs.
        _, res = self._fit(0)
        rmse = float(np.sqrt(np.mean((res.h_hat - np.sin(self.grid)) ** 2)))
        assert rmse < 0.12

    def test_beats_the_conditional_mean_it_is_meant_to_correct(self):
        # E[y | d] = sin(d) + E[u | d] = sin(d) + 0.5 d / 1.89. A method
        # that returns the regression function is off by that second term.
        df, res = self._fit(1)
        d = df["d"].to_numpy()
        y = df["y"].to_numpy()
        h = 0.25
        w = np.exp(-0.5 * ((self.grid[:, None] - d[None, :]) / h) ** 2)
        naive = (w @ y) / w.sum(axis=1)
        truth = np.sin(self.grid)
        err_iv = float(np.sqrt(np.mean((res.h_hat - truth) ** 2)))
        err_naive = float(np.sqrt(np.mean((naive - truth) ** 2)))
        assert err_naive > 0.2
        assert err_iv < 0.5 * err_naive

    def test_uniform_band_contains_the_truth(self):
        _, res = self._fit(2)
        truth = np.sin(self.grid)
        assert np.all(res.ci_low <= truth)
        assert np.all(truth <= res.ci_high)
        # Not by being vacuous: the band is narrower than the confounding
        # bias it has to exclude at the edge of the grid.
        assert float(np.max(res.ci_high - res.ci_low)) < 0.8


# --------------------------------------------------------------------- #
#  bcf_ordinal: dose-level effects and their standard errors
# --------------------------------------------------------------------- #


class TestBCFOrdinalStandardErrors:
    @pytest.fixture(scope="class")
    def fit(self):
        rng = np.random.default_rng(0)
        n = 1500
        X = rng.normal(size=(n, 3))
        T = rng.integers(0, 4, size=n)
        Y = X[:, 0] + 0.5 * T + 0.3 * X[:, 0] * T + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"Y": Y, "T": T, "X0": X[:, 0], "X1": X[:, 1], "X2": X[:, 2]})
        return sp.bcf_ordinal(
            df, y="Y", treat="T", covariates=["X0", "X1", "X2"], n_bootstrap=30
        )

    def test_dose_effects(self, fit):
        ate = np.asarray(fit.ate, dtype=float)
        # Across-seed SD at this n: 0.03, 0.05, 0.05.
        np.testing.assert_allclose(ate, [0.5, 1.0, 1.5], atol=0.15)

    def test_standard_errors_are_on_the_scale_of_the_sampling_error(self, fit):
        # The SE used to be sqrt(mean CATE variance / n), about 0.001,
        # against an across-seed SD of 0.03 to 0.05. It is now the root of
        # the summed adjacent-step variances: 0.024, 0.034, 0.042.
        se = np.asarray(fit.ate_se, dtype=float)
        assert np.all(se > 0.012)
        assert np.all(se < 0.10)
        ate = np.asarray(fit.ate, dtype=float)
        assert np.all(np.abs(ate - [0.5, 1.0, 1.5]) <= 4.0 * se)

    def test_standard_errors_grow_with_the_number_of_steps(self, fit):
        se = np.asarray(fit.ate_se, dtype=float)
        assert se[0] < se[1] < se[2]


# --------------------------------------------------------------------- #
#  bcf_longitudinal: the second pass no longer inflates the effect
# --------------------------------------------------------------------- #


def _randomised_panel(seed: int, n: int = 100, periods: int = 3, tau: float = 6.0):
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 0.5, n)
    parts = []
    for t in range(periods):
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        d = rng.integers(0, 2, n)
        y = 0.5 * x1 + 0.3 * x2 + tau * d + u + 0.2 * t + rng.normal(0, 0.5, n)
        parts.append(
            pd.DataFrame(
                {"id": np.arange(n), "t": t, "x1": x1, "x2": x2, "d": d, "y": y}
            )
        )
    return pd.concat(parts, ignore_index=True)


class TestBCFLongitudinal:
    kw = dict(
        outcome="y",
        treatment="d",
        unit="id",
        time="t",
        covariates=["x1", "x2"],
        n_bootstrap=20,
        n_trees_mu=60,
        n_trees_tau=30,
    )

    def test_constant_effect_under_randomisation(self):
        # Treatment is a coin flip each period, so the answer is 6.0 and a
        # difference in means already gets it. The previous version gave
        # 7.44, 7.77 and 7.61 on seeds 0 to 2: an in-sample propensity
        # score and a residual that left -e * tau in the unit effects
        # pushed the second pass up by a quarter of the effect.
        df = _randomised_panel(0)
        res = sp.bcf_longitudinal(df, random_state=0, **self.kw)
        assert res.average_ate == pytest.approx(6.0, abs=0.3)
        np.testing.assert_allclose(
            res.per_time_ate["ate_point"].to_numpy(), 6.0, atol=0.5
        )
        lo, hi = res.average_ci
        assert lo < 6.0 < hi

    def test_period_without_treatment_variation_is_not_counted_as_zero(self):
        df = _randomised_panel(1)
        # Nobody is treated in the first period: take the effect back out
        # of the outcome and clear the indicator.
        first_period = df["t"] == 0
        df.loc[first_period, "y"] -= 6.0 * df.loc[first_period, "d"]
        df.loc[first_period, "d"] = 0
        with pytest.warns(AssumptionWarning, match="No treatment variation"):
            res = sp.bcf_longitudinal(df, random_state=1, **self.kw)
        first = res.per_time_ate.set_index("time").loc[0, "ate_point"]
        assert np.isnan(first)
        # Two identified periods at 6.0. Averaging in a zero for the
        # third would give 4.0.
        assert res.average_ate == pytest.approx(6.0, abs=0.4)

    def test_no_identified_period_raises(self):
        df = _randomised_panel(2)
        df["d"] = 0
        with pytest.raises(ValueError, match="not identified"):
            sp.bcf_longitudinal(df, random_state=2, **self.kw)


# --------------------------------------------------------------------- #
#  Recoveries
# --------------------------------------------------------------------- #


def test_conformal_interference_covers_held_out_cluster_means():
    # Nominal 90%. Across 10 seeds the mean coverage was 0.93.
    cover = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        rows = []
        for c in range(80):
            shock = rng.normal(0, 0.5)
            for _ in range(10):
                tr = rng.integers(0, 2)
                x = rng.normal()
                y = 1.0 + 0.6 * tr + 0.4 * x + shock + rng.normal(0, 0.5)
                rows.append({"cl": c, "d": tr, "x": x, "y": y})
        df = pd.DataFrame(rows)
        test = list(range(50, 80))
        res = sp.conformal_interference(
            df,
            y="y",
            treatment="d",
            cluster="cl",
            covariates=["x"],
            test_clusters=test,
            alpha=0.1,
            random_state=seed,
        )
        pred = res.predictions.set_index("cluster")
        ybar = df[df["cl"].isin(test)].groupby("cl")["y"].mean()
        lo = pred["lo"].reindex(ybar.index)
        hi = pred["hi"].reindex(ybar.index)
        cover.append(float(np.mean((lo <= ybar) & (ybar <= hi))))
    assert 0.82 <= float(np.mean(cover)) <= 1.0


def _bandit_log(seed: int = 0, n: int = 6000):
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 3, size=n)
    a = rng.integers(0, 2, size=n)
    return rng, s, a


def test_causal_dqn_learns_the_rewarded_action():
    # Reward 1 when the action equals state mod 2, transitions uniform.
    rng, s, a = _bandit_log()
    r = (a == s % 2).astype(float) + rng.normal(0, 0.1, size=len(s))
    s_next = rng.integers(0, 3, size=len(s))
    df = pd.DataFrame({"s": s, "a": a, "r": r, "s_next": s_next})
    res = sp.causal_dqn(df, state="s", action="a", reward="r", next_state="s_next")
    np.testing.assert_array_equal(res.policy, [0, 1, 0])
    gap = np.abs(res.q_table[:, 0] - res.q_table[:, 1])
    # One step of reward is worth 1; the bound shaves a little off.
    assert np.all(gap > 0.5)


class TestOfflineSafePolicy:
    @pytest.fixture(scope="class")
    def log(self):
        rng, s, a = _bandit_log()
        reward = (a == s % 2) * 1.0 + rng.normal(0, 0.3, len(s))
        cost = a * 0.6 + rng.normal(0, 0.05, len(s))
        return pd.DataFrame({"state": s, "action": a, "reward": reward, "cost": cost})

    def _fit(self, log, threshold):
        return sp.offline_safe_policy(
            log,
            state="state",
            action="action",
            reward="reward",
            cost="cost",
            cost_threshold=threshold,
        )

    def test_binding_constraint_rules_out_the_costly_action(self, log):
        # Action 1 costs 0.6 each time; a budget of 0.3 leaves action 0.
        res = self._fit(log, 0.3)
        np.testing.assert_array_equal(res.policy, [0, 0, 0])
        assert res.feasible
        assert res.expected_cost <= 0.3
        # Action 0 is right in two states out of three.
        assert res.expected_reward == pytest.approx(2 / 3, abs=0.05)

    def test_slack_constraint_returns_the_unconstrained_optimum(self, log):
        res = self._fit(log, 0.9)
        np.testing.assert_array_equal(res.policy, [0, 1, 0])
        assert res.expected_reward == pytest.approx(1.0, abs=0.05)
        assert res.expected_cost == pytest.approx(0.2, abs=0.05)


def _text_confounder_design(seed: int, n: int = 800) -> pd.DataFrame:
    """The words carry a confounder: effect 1.0, raw contrast 1.8."""
    good = ["great", "love", "excellent", "fine"]
    bad = ["terrible", "slow", "buggy", "bad"]
    rng = np.random.default_rng(seed)
    c = rng.integers(0, 2, n)
    t = (rng.uniform(size=n) < np.where(c == 1, 0.7, 0.3)).astype(int)
    text = [" ".join(rng.choice(good if ci else bad, 4)) for ci in c]
    y = 1.0 * t + 2.0 * c + rng.normal(0, 0.5, n)
    return pd.DataFrame({"text": text, "t": t, "y": y})


def test_text_treatment_effect_removes_text_borne_confounding():
    # 30 seeds at 256 buckets: mean 1.018, across-seed SD 0.040, mean SE
    # 0.049, every interval covering.
    est, se, raw = [], [], []
    for seed in range(4):
        df = _text_confounder_design(seed)
        # Experimental API: getattr keeps this anchor from promoting it in
        # the parity index.
        res = getattr(sp, "text_treatment_effect")(
            df,
            text_col="text",
            outcome="y",
            treatment="t",
            n_components=256,
            seed=seed,
        )
        est.append(float(res.estimate))
        se.append(float(res.se))
        raw.append(df.loc[df["t"] == 1, "y"].mean() - df.loc[df["t"] == 0, "y"].mean())
        assert abs(est[-1] - 1.0) <= 4.0 * se[-1]
    assert np.mean(raw) > 1.6
    assert np.mean(est) == pytest.approx(1.0, abs=0.08)
    assert 0.03 < np.mean(se) < 0.08


def test_text_treatment_effect_bucket_collisions_leave_confounding():
    # The documented limit of the hash embedder: with 20 buckets some of
    # the eight words collide for most salts and the adjustment is
    # partial. Mean 1.076 over 30 seeds, across-seed SD twice the SE.
    est = []
    for seed in range(12):
        df = _text_confounder_design(seed)
        res = getattr(sp, "text_treatment_effect")(
            df, text_col="text", outcome="y", treatment="t", n_components=20, seed=seed
        )
        est.append(float(res.estimate))
    assert np.mean(est) > 1.03
    assert np.max(est) > 1.15


def test_iv_bounds_run_from_the_wald_ratio_to_the_ols_contrast():
    # With a valid randomised instrument the Wald ratio is the complier
    # effect (0.117 here, by simulation of the complier population) and
    # the raw contrast is confounded upward. The bounds are the interval
    # between the two.
    lower, upper = [], []
    for seed in range(6):
        rng = np.random.default_rng(seed)
        n = 4000
        z = rng.binomial(1, 0.5, n)
        u = rng.normal(size=n)
        d = ((z + u) > 0.5).astype(int)
        y = (0.3 * d + 0.5 * u + rng.normal(size=n) > 0).astype(int)
        df = pd.DataFrame({"y": y, "d": d, "z": z})
        res = sp.iv_bounds(df, y="y", treatment="d", instrument="z", n_boot=30)
        wald = res.model_info["wald_estimate"]
        ols = y[d == 1].mean() - y[d == 0].mean()
        assert res.lower == pytest.approx(min(wald, ols), abs=1e-12)
        assert res.upper == pytest.approx(max(wald, ols), abs=1e-12)
        lower.append(res.lower)
        upper.append(res.upper)
    # Wald SE at this n is about 0.04, so the mean of six has SE 0.017.
    assert np.mean(lower) == pytest.approx(0.117, abs=0.06)
    assert np.mean(upper) > 0.3


def test_rd_bayes_hte_posterior_at_the_cutoff():
    # Jump 0.4 + 0.3 z with z standard normal: average jump 0.4, unit
    # effects spread with SD 0.3.
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.uniform(-1, 1, n)
    z = rng.normal(size=n)
    y = 1 + 0.5 * x + (0.4 + 0.3 * z) * (x >= 0) + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": y, "run": x, "z": z})
    res = sp.rd_bayes_hte(df, y="y", running="run", covariates=["z"])
    assert abs(res.posterior_mean - 0.4) <= 4.0 * res.posterior_sd
    lo, hi = res.posterior_ci
    assert lo < 0.4 < hi
    slope = np.polyfit(z, res.cate, 1)[0]
    assert slope == pytest.approx(0.3, abs=0.08)


def test_compare_metalearners_share_one_ate_and_differ_in_cate():
    # Every learner reports the same cross-fitted AIPW average effect by
    # design; the learner only shapes the CATE. At n = 2000 the SE (0.057)
    # matches the across-seed SD (0.055) on this design.
    rng = np.random.default_rng(0)
    n = 2000
    age = rng.normal(size=n)
    edu = rng.normal(size=n)
    tr = rng.integers(0, 2, size=n)
    wage = 0.5 * age + 0.3 * edu + (1.0 + age) * tr + rng.normal(size=n)
    df = pd.DataFrame({"wage": wage, "training": tr, "age": age, "edu": edu})
    table = sp.compare_metalearners(
        df, y="wage", treat="training", covariates=["age", "edu"]
    )
    assert table["ate"].nunique() == 1
    assert table["se"].nunique() == 1
    ate, se = float(table["ate"].iloc[0]), float(table["se"].iloc[0])
    assert abs(ate - 1.0) <= 4.0 * se
    assert 0.03 < se < 0.09
    assert table["cate_std"].nunique() == len(table)
