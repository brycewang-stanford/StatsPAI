"""Third batch of known-truth anchors from the 2026-10 pass.

Fixes. ``sp.kernel_iv`` (the structural function under confounding),
``sp.bcf_ordinal`` (standard error of the dose-level effects),
``sp.bcf_longitudinal`` (second-pass bias; periods with no treatment
variation), and two simulators whose data did not carry the effect they
declared: ``sp.dgp_rd(fuzzy=True)`` and ``sp.dgp_rdit``.

Simulators and utilities. Each ``sp.dgp_*`` design checked against the
estimator that is unbiased for it, and the row / rank helpers checked
against pandas.

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


# --------------------------------------------------------------------- #
#  Simulators: the declared truth is the truth
# --------------------------------------------------------------------- #


def _mean_and_se(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


class TestRDSimulator:
    def test_sharp_default_draws_are_unchanged(self):
        # The fix to the fuzzy branch and to `cutoff` must not move the
        # default sharp design: same generator calls, same order.
        rng = np.random.default_rng(3)
        x = rng.uniform(-1, 1, size=500)
        y = 0.5 * x + 0.3 * x**2 + 0.3 * (x >= 0) + rng.normal(0, 0.3, size=500)
        df = sp.dgp_rd(n=500, seed=3)
        np.testing.assert_array_equal(df["x"].to_numpy(), x)
        np.testing.assert_array_equal(df["y"].to_numpy(), y)

    def test_fuzzy_design_has_a_first_stage_jump(self):
        # It used to draw treatment from a logistic curve that is smooth
        # through the cutoff, which is not a fuzzy RD and identifies
        # nothing: fuzzy estimates had a median of 0.34 and no finite mean.
        df = sp.dgp_rd(n=40000, fuzzy=True, seed=0)
        assert df.attrs["first_stage_jump"] == 0.6
        near = df[df["x"].abs() < 0.05]
        jump = (
            near.loc[near["x"] >= 0, "treatment"].mean()
            - near.loc[near["x"] < 0, "treatment"].mean()
        )
        assert jump == pytest.approx(0.6, abs=0.05)

    def test_fuzzy_design_delivers_the_declared_effect(self):
        est = []
        for seed in range(8):
            df = sp.dgp_rd(n=6000, effect=0.3, fuzzy=True, seed=seed)
            res = sp.rdrobust(df, y="y", x="x", c=0.0, fuzzy="treatment")
            est.append(res.estimate)
        mean, se = _mean_and_se(est)
        # 60 seeds: 0.301 with a standard error of the mean of 0.009.
        assert abs(mean - 0.3) <= 4.0 * se
        assert se < 0.05

    def test_cutoff_and_spread_move_the_running_variable(self):
        # `cutoff=5` used to leave x on (-1, 1): nobody treated.
        df = sp.dgp_rd(n=6000, effect=0.3, cutoff=5.0, bandwidth_relevant=2.0, seed=0)
        assert df["x"].min() == pytest.approx(1.0, abs=0.01)
        assert df["x"].max() == pytest.approx(9.0, abs=0.01)
        assert df["treatment"].mean() == pytest.approx(0.5, abs=0.03)
        res = sp.rdrobust(df, y="y", x="x", c=5.0)
        assert abs(res.estimate - 0.3) <= 4.0 * res.se

    def test_spread_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            sp.dgp_rd(n=100, bandwidth_relevant=0.0, seed=0)


def test_rdit_simulator_level_shift_is_the_declared_effect():
    # The AR(1) recursion used to run on the outcome, so the shift fed
    # back into itself and settled at effect / 0.7 = 2.86. A regression
    # on a step and two linear trends gave 2.81 over 60 seeds.
    est = []
    for seed in range(40):
        df = sp.dgp_rdit(
            n_periods=400, effect=2.0, cutoff_period=200, seasonality=False, seed=seed
        )
        post = (df["time"] >= 200).to_numpy(dtype=float)
        tc = (df["time"] - 200).to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(df)), post, tc, post * tc])
        beta = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)[0]
        est.append(beta[1])
    mean, se = _mean_and_se(est)
    assert abs(mean - 2.0) <= 4.0 * se
    assert se < 0.04


def test_rdit_simulator_errors_are_ar1():
    # 3000 monthly periods is the most the date index can hold.
    df = sp.dgp_rdit(n_periods=3000, effect=0.0, seasonality=False, seed=0)
    e = df["y"].to_numpy() - 0.01 * df["time"].to_numpy()
    rho = np.corrcoef(e[1:], e[:-1])[0, 1]
    assert rho == pytest.approx(0.3, abs=0.07)
    # Stationary variance 0.25 / (1 - 0.09).
    assert e.var() == pytest.approx(0.25 / 0.91, rel=0.15)


def _ols_coef(df: pd.DataFrame, y: str, target: str, others=(), absorb=()):
    cols = [df[target].to_numpy(dtype=float)]
    cols += [df[c].to_numpy(dtype=float) for c in others]
    for a in absorb:
        cols += list(pd.get_dummies(df[a], drop_first=True).to_numpy(dtype=float).T)
    X = np.column_stack([np.ones(len(df))] + cols)
    return float(np.linalg.lstsq(X, df[y].to_numpy(dtype=float), rcond=None)[0][1])


@pytest.mark.parametrize(
    "name, draw, estimate, truth",
    [
        (
            "rct",
            lambda s: sp.dgp_rct(n=2000, effect=0.3, seed=s),
            lambda d: _ols_coef(d, "y", "treatment"),
            0.3,
        ),
        (
            "rct, heterogeneous",
            lambda s: sp.dgp_rct(n=2000, effect=0.3, heterogeneous=True, seed=s),
            lambda d: _ols_coef(d, "y", "treatment"),
            0.3,
        ),
        (
            "observational, adjusted",
            lambda s: sp.dgp_observational(n=4000, effect=0.5, seed=s),
            lambda d: _ols_coef(d, "y", "treatment", others=["x1", "x2"]),
            0.5,
        ),
        (
            "panel, two-way FE",
            lambda s: sp.dgp_panel(n_units=60, n_periods=20, seed=s),
            lambda d: _ols_coef(d, "y", "x", absorb=["unit", "time"]),
            1.0,
        ),
        (
            "did, two-way FE",
            lambda s: sp.dgp_did(n_units=120, n_periods=10, effect=0.5, seed=s),
            lambda d: _ols_coef(d, "y", "treated", absorb=["unit", "time"]),
            0.5,
        ),
        (
            "cluster rct",
            lambda s: sp.dgp_cluster_rct(n_clusters=200, cluster_size=20, seed=s),
            lambda d: _ols_coef(d, "y", "treatment"),
            0.3,
        ),
        (
            "bartik",
            lambda s: sp.dgp_bartik(n_regions=400, effect=1.0, seed=s)["data"],
            lambda d: _ols_coef(d, "y", "bartik"),
            1.0,
        ),
    ],
)
def test_simulator_delivers_its_declared_effect(name, draw, estimate, truth):
    est = [estimate(draw(seed)) for seed in range(24)]
    mean, se = _mean_and_se(est)
    assert abs(mean - truth) <= 4.0 * se, name
    assert se < 0.05, name


def test_observational_simulator_is_actually_confounded():
    raw = []
    for seed in range(8):
        d = sp.dgp_observational(n=4000, effect=0.5, seed=seed)
        raw.append(_ols_coef(d, "y", "treatment"))
    assert np.mean(raw) > 0.6


def test_iv_simulator_is_endogenous_and_the_instrument_fixes_it():
    iv, ols = [], []
    for seed in range(12):
        d = sp.dgp_iv(n=4000, effect=0.5, seed=seed)
        res = sp.ivreg("y ~ x1 + x2 + (treatment ~ instrument)", data=d)
        iv.append(float(res.params["treatment"]))
        ols.append(_ols_coef(d, "y", "treatment", others=["x1", "x2"]))
    mean, se = _mean_and_se(iv)
    assert abs(mean - 0.5) <= 4.0 * se
    assert np.mean(ols) > 0.7


def test_cluster_rct_simulator_has_the_declared_icc():
    sb, sw = [], []
    for seed in range(12):
        d = sp.dgp_cluster_rct(n_clusters=200, cluster_size=20, icc=0.1, seed=seed)
        c = d[d["treatment"] == 0]
        within = c.groupby("cluster_id")["y"].var().mean()
        between = c.groupby("cluster_id")["y"].mean().var() - within / 20
        sb.append(between)
        sw.append(within)
    icc = np.mean(sb) / (np.mean(sb) + np.mean(sw))
    assert icc == pytest.approx(0.1, abs=0.02)


def test_bartik_simulator_instrument_is_the_share_weighted_shock():
    d = sp.dgp_bartik(n_regions=50, n_industries=10, seed=0)
    np.testing.assert_allclose(
        np.asarray(d["shares"]) @ np.asarray(d["shocks"]),
        d["data"]["bartik"].to_numpy(),
        atol=1e-12,
    )


def test_bunching_simulator_matches_the_iso_elastic_model():
    df = sp.dgp_bunching(n=100000, kink_point=50000.0, elasticity=0.3, seed=0)
    upper = 50000.0 * (1.0 / 0.8) ** 0.3
    assert df.attrs["bunching_upper"] == pytest.approx(upper, rel=1e-12)
    cf = df["counterfactual_income"].to_numpy()
    inc = df["income"].to_numpy()
    below = cf <= 50000.0
    bunch = (cf > 50000.0) & (cf < upper)
    above = cf >= upper
    np.testing.assert_array_equal(inc[below], cf[below])
    np.testing.assert_allclose(inc[bunch], 50000.0)
    np.testing.assert_allclose(inc[above], cf[above] * 0.8**0.3, rtol=1e-12)
    assert df.attrs["n_bunchers"] == int(bunch.sum())


# --------------------------------------------------------------------- #
#  Data utilities: identities against pandas / numpy
# --------------------------------------------------------------------- #


class TestRowAndRankUtilities:
    @pytest.fixture(scope="class")
    def frame(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.normal(size=(50, 3)), columns=list("abc"))
        df.loc[3, "b"] = np.nan
        df["g"] = rng.integers(0, 3, 50)
        df["v"] = rng.integers(0, 10, 50).astype(float)
        return df

    @pytest.mark.parametrize(
        "name, reference",
        [
            ("rowmean", lambda d: d.mean(axis=1)),
            ("rowtotal", lambda d: d.sum(axis=1)),
            ("rowmax", lambda d: d.max(axis=1)),
            ("rowmin", lambda d: d.min(axis=1)),
            ("rowsd", lambda d: d.std(axis=1, ddof=1)),
        ],
    )
    def test_row_functions_skip_missing_like_stata_egen(self, frame, name, reference):
        got = getattr(sp, name)(frame, list("abc"))
        np.testing.assert_allclose(got, reference(frame[list("abc")]), atol=1e-14)

    def test_rowcount_counts_non_missing(self, frame):
        got = getattr(sp, "rowcount")(frame, list("abc"))
        assert got.iloc[3] == 2
        assert (got.drop(index=3) == 3).all()

    def test_rank_average_ties_overall_and_by_group(self, frame):
        np.testing.assert_allclose(sp.rank(frame, "v"), frame["v"].rank())
        np.testing.assert_allclose(
            sp.rank(frame, "v", by="g"), frame.groupby("g")["v"].rank()
        )

    def test_pwcorr_is_pairwise_complete(self, frame):
        got = sp.pwcorr(frame, vars=list("abc"), output="dataframe", stars=False)
        want = frame[list("abc")].corr()
        np.testing.assert_allclose(
            got.to_numpy(dtype=float), want.to_numpy(), atol=5e-4
        )

    def test_outlier_indicator_flags_outside_the_percentiles(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"v": rng.normal(size=1000)})
        out = sp.outlier_indicator(df, ["v"], cuts=(1, 99))
        lo, hi = np.percentile(df["v"], [1, 99])
        want = ((df["v"] < lo) | (df["v"] > hi)).to_numpy()
        np.testing.assert_array_equal(out["v_outlier"].to_numpy().astype(bool), want)


def test_scalar_iv_projection_is_the_first_stage_fitted_value():
    rng = np.random.default_rng(0)
    n = 500
    z1, z2, x = rng.normal(size=(3, n))
    d = 0.5 * z1 + 0.3 * z2 + 0.2 * x + rng.normal(size=n)
    df = pd.DataFrame({"d": d, "z1": z1, "z2": z2, "x": x})
    got = sp.scalar_iv_projection(
        df, "d", ["z1", "z2"], covariates=["x"], return_column=True
    )
    X = np.column_stack([np.ones(n), z1, z2, x])
    fitted = X @ np.linalg.lstsq(X, d, rcond=None)[0]
    np.testing.assert_allclose(got, fitted, atol=1e-12)


# --------------------------------------------------------------------- #
#  conformal_ite_multidp: the interval is for an effect, not an outcome
# --------------------------------------------------------------------- #


def _two_stage_potential_outcomes(seed: int, dependence: str, n: int = 1500):
    """Two randomised stages; returns the data and the true unit effects."""
    rng = np.random.default_rng(seed)
    frames, effects = {}, []
    x_prev = None
    for k, (base, slope, tau) in enumerate([(1.0, 0.5, 0.8), (0.5, 0.3, 0.6)], 1):
        x = rng.normal(size=n) if x_prev is None else 0.5 * x_prev + rng.normal(size=n)
        d = rng.binomial(1, 0.5, n)
        e0 = rng.normal(0, 0.5, n)
        e1 = rng.normal(0, 0.5, n) if dependence == "independent" else -e0
        y0 = base + slope * x + e0
        y1 = base + slope * x + tau + e1
        frames[f"x{k}"], frames[f"d{k}"] = x, d
        frames[f"y{k}"] = np.where(d == 1, y1, y0)
        effects.append(y1 - y0)
        x_prev = x
    return pd.DataFrame(frames), effects


@pytest.mark.parametrize("dependence", ["independent", "opposed"])
def test_conformal_ite_multidp_covers_the_individual_effects(dependence):
    # alpha = 0.1 over two stages: each stage owes 95%, both together 90%.
    # The half-width used to be one quantile of single-outcome residuals,
    # which covers a potential outcome and not a difference of two: 83%
    # per stage with independent noise. With one quantile per arm the
    # union bound holds for any dependence; opposed noise is the case
    # that comes closest to using it up (97.5% per stage in theory).
    per_stage, joint = [], []
    for seed in range(6):
        df, effects = _two_stage_potential_outcomes(seed, dependence)
        res = sp.conformal_ite_multidp(
            df.iloc[:1000],
            y_per_stage=["y1", "y2"],
            treat_per_stage=["d1", "d2"],
            history_per_stage=[["x1"], ["x1", "x2"]],
            test_data=df.iloc[1000:],
            alpha=0.1,
            seed=seed,
        )
        inside = []
        for interval, effect in zip(res.intervals_per_stage, effects):
            truth = effect[1000:]
            inside.append((interval[:, 0] <= truth) & (truth <= interval[:, 1]))
            per_stage.append(inside[-1].mean())
        joint.append((inside[0] & inside[1]).mean())
        total = effects[0][1000:] + effects[1][1000:]
        cum = res.cumulative_interval
        assert np.mean((cum[:, 0] <= total) & (total <= cum[:, 1])) >= 0.9
    assert np.mean(per_stage) >= 0.95
    assert np.mean(joint) >= 0.90
    if dependence == "opposed":
        # Not vacuous: the worst case sits within a few points of nominal.
        assert np.mean(per_stage) <= 0.995


def test_conformal_ite_multidp_reports_a_calibration_set_that_is_too_small():
    df, _ = _two_stage_potential_outcomes(0, "independent", n=40)
    with pytest.warns(RuntimeWarning, match="cannot support level"):
        sp.conformal_ite_multidp(
            df,
            y_per_stage=["y1", "y2"],
            treat_per_stage=["d1", "d2"],
            history_per_stage=[["x1"], ["x1", "x2"]],
            alpha=0.1,
        )


# --------------------------------------------------------------------- #
#  deepiv: the default loss is the consistent one
# --------------------------------------------------------------------- #


def _linear_iv_design(seed: int, n: int = 3000) -> pd.DataFrame:
    """y = d + 0.5 x + u; the instrument explains 56% of d given x."""
    rng = np.random.default_rng(seed)
    z, x, u = rng.normal(size=(3, n))
    d = 0.8 * z + 0.5 * u + 0.3 * x + rng.normal(0, 0.5, n)
    y = 1.0 * d + 0.5 * x + u + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, "d": d, "z": z, "x": x})


class TestDeepIVLoss:
    kw = dict(y="y", treat="d", instruments=["z"], covariates=["x"])

    @staticmethod
    def _per_unit(res) -> float:
        # The estimate is the effect of a one-SD rise in the treatment.
        return float(res.estimate / res.model_info["treatment_shift"])

    def test_default_recovers_a_linear_structural_slope(self):
        pytest.importorskip("torch")
        # Eight seeds: 0.90 to 1.14, mean 1.04. OLS gives 1.53.
        slopes = []
        for seed in (0, 1):
            df = _linear_iv_design(seed)
            res = sp.deepiv(df, random_state=seed, **self.kw)
            assert res.model_info["gradient_estimator"] == "unbiased (paired)"
            slopes.append(self._per_unit(res))
        assert 0.85 < np.mean(slopes) < 1.25

    def test_single_sample_loss_is_attenuated_and_says_so(self):
        pytest.importorskip("torch")
        # The former default. Its limit on this design is the first-stage
        # share 0.64 / (0.64 + 0.50) = 0.56; three seeds gave 0.50 to 0.71.
        df = _linear_iv_design(1)
        with pytest.warns(AssumptionWarning, match="upper bound"):
            res = getattr(sp, "deepiv")(
                df, n_gradient_samples=0, random_state=1, **self.kw
            )
        assert self._per_unit(res) < 0.8

    def test_fixed_network_se_is_flagged(self):
        pytest.importorskip("torch")
        df = _linear_iv_design(2, n=600)
        res = sp.deepiv(
            df,
            first_stage_epochs=10,
            second_stage_epochs=10,
            random_state=2,
            **self.kw,
        )
        assert res.model_info["se_valid_for_ate"] is False


# --------------------------------------------------------------------- #
#  vcnet / scigan: the curve is read off the basis it was fitted on
# --------------------------------------------------------------------- #


def _dose_design(seed: int, n: int = 2000) -> pd.DataFrame:
    """E[Y(t)] = 1 + 2 t - t^2; the dose rises with the confounder x."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = np.clip(0.5 + 0.2 * x + rng.normal(0, 0.2, n), 0, 1)
    y = 1 + 2 * t - t**2 + 0.5 * x + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, "t": t, "x": x})


class TestVCNetCurve:
    grid = [0.2, 0.5, 0.8]
    truth = np.array([1.36, 1.75, 1.96])

    def test_curve_on_an_interior_grid(self):
        # The knots for the evaluation grid used to be laid on the grid's
        # own range, so the first point returned the curve at the smallest
        # observed dose and the last at the largest: 1.01 and 1.99 here
        # (the truth at t = 0 and t = 1), 0% coverage at t = 0.2.
        # 60 seeds now: 1.354, 1.752, 1.963; SE equal to the across-seed
        # SD; 93% coverage at each point.
        df = _dose_design(0)
        res = sp.vcnet(
            df, y="y", treatment="t", covariates=["x"], t_grid=self.grid, n_bootstrap=60
        )
        assert np.all(np.abs(res.mu_hat - self.truth) <= 4.0 * res.se)
        assert np.all(res.se < 0.06)
        # A quadratic in t alone is confounded by x: 0.97 and 2.36.
        raw = np.polyval(np.polyfit(df["t"], df["y"], 2), self.grid)
        assert abs(raw[0] - self.truth[0]) > 0.25
        assert abs(raw[2] - self.truth[2]) > 0.25

    def test_default_grid_agrees_with_a_grid_given_by_hand(self):
        df = _dose_design(1)
        kw = dict(y="y", treatment="t", covariates=["x"], n_bootstrap=20)
        auto = sp.vcnet(df, **kw)
        by_hand = sp.vcnet(df, t_grid=auto.t_grid[5:30], **kw)
        np.testing.assert_allclose(by_hand.mu_hat, auto.mu_hat[5:30], atol=1e-10)

    def test_se_includes_the_average_over_covariates(self):
        # At t = 0.5 the spline part is tight and most of the sampling
        # error is 0.5 * mean(x): SD 0.5 / sqrt(2000) = 0.011. Holding the
        # covariate mean fixed across bootstrap draws gave 0.012 against
        # an across-seed SD of 0.017.
        est, se = [], []
        for seed in range(12):
            res = sp.vcnet(
                _dose_design(seed),
                y="y",
                treatment="t",
                covariates=["x"],
                t_grid=[0.5],
                n_bootstrap=60,
                random_state=seed,
            )
            est.append(res.mu_hat[0])
            se.append(res.se[0])
        ratio = np.mean(se) / np.std(est, ddof=1)
        assert 0.65 < ratio < 1.6
        assert np.mean(se) > 0.0135

    def test_grid_outside_the_observed_doses_is_not_extrapolated(self):
        df = _dose_design(2)
        with pytest.warns(AssumptionWarning, match="outside the observed dose"):
            res = sp.vcnet(
                df,
                y="y",
                treatment="t",
                covariates=["x"],
                t_grid=[-0.5, 0.5, 1.5],
                n_bootstrap=20,
            )
        assert np.isnan(res.mu_hat[0]) and np.isnan(res.mu_hat[2])
        assert res.mu_hat[1] == pytest.approx(1.75, abs=0.08)


class TestSciganWeights:
    kw = dict(y="y", treatment="t", covariates=["x"], t_grid=[0.2, 0.5, 0.8])

    def test_unit_weights_reproduce_vcnet(self):
        # The weights used to be applied by drawing a weighted resample of
        # the rows, so unit weights returned a noisier vcnet.
        df = _dose_design(3)
        a = sp.vcnet(df, n_bootstrap=20, **self.kw)
        b = sp.scigan(
            df, propensity_weights=np.ones(len(df)), n_bootstrap=20, **self.kw
        )
        np.testing.assert_array_equal(a.mu_hat, b.mu_hat)
        np.testing.assert_array_equal(a.se, b.se)

    def test_integer_weights_equal_row_duplication(self):
        df = _dose_design(4, n=600)
        w = np.random.default_rng(0).integers(1, 4, len(df))
        weighted = sp.scigan(df, propensity_weights=w, n_bootstrap=20, **self.kw)
        dup = df.loc[df.index.repeat(w)].reset_index(drop=True)
        # The weights are normalised to sum to n, so the ridge penalty of
        # the duplicated fit is scaled to match.
        expanded = sp.vcnet(
            dup, ridge=1e-2 * len(dup) / len(df), n_bootstrap=20, **self.kw
        )
        np.testing.assert_allclose(weighted.mu_hat, expanded.mu_hat, atol=1e-10)


def test_gnn_causal_adjusts_for_neighbour_confounding():
    # Treatment and outcome both load on the neighbours' mean covariate.
    # 30 seeds: mean 1.01, SE 0.050 against an across-seed SD of 0.048,
    # 97% coverage; the raw contrast is 1.83.
    est, se, raw = [], [], []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n = 600
        A = np.triu((rng.uniform(size=(n, n)) < 0.015).astype(float), 1)
        A = A + A.T
        x = rng.normal(size=n)
        xn = (A @ x) / np.maximum(A.sum(1), 1)
        d = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x - 0.8 * xn)))
        y = 1 + x + 1.5 * xn + 1.0 * d + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"y": y, "d": d, "x": x})
        res = sp.gnn_causal(
            df, y="y", treat="d", covariates=["x"], adjacency=A, random_state=seed
        )
        est.append(res.ate)
        se.append(res.se)
        raw.append(y[d == 1].mean() - y[d == 0].mean())
    assert np.mean(raw) > 1.5
    assert abs(np.mean(est) - 1.0) <= 4.0 * np.mean(se) / np.sqrt(len(est))
    assert 0.03 < np.mean(se) < 0.08


@pytest.mark.parametrize("name", ["tarnet", "cfrnet", "dragonnet"])
def test_neural_outcome_models_remove_observed_confounding(name):
    pytest.importorskip("torch")
    # ATE 1.0 with a heterogeneous effect; the raw contrast is 1.65.
    # Four seeds each: 0.96 to 1.03. Only dragonnet's SE is a standard
    # error for the ATE (0.026, on the scale of the efficiency bound);
    # the other two flag theirs in model_info.
    rng = np.random.default_rng(0)
    n = 2000
    x1, x2 = rng.normal(size=(2, n))
    d = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x1)))
    y = 1 + x1 + 0.5 * x2 + (1.0 + 0.5 * x2) * d + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
    res = getattr(sp, name)(
        df, y="y", treat="d", covariates=["x1", "x2"], epochs=150, n_bootstrap=50
    )
    assert y[d == 1].mean() - y[d == 0].mean() > 1.5
    assert res.estimate == pytest.approx(1.0, abs=0.12)
    if name == "dragonnet":
        assert 0.015 < res.se < 0.05
    else:
        assert res.model_info["se_valid_for_ate"] is False
