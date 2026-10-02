"""Second batch of known-truth anchors from the 2026-10 pass.

Three groups.

Fixes. ``sp.event_study`` (never-treated coding and event times with no
treated observation), the power calculators ``sp.power_did`` /
``sp.power_rd`` / ``sp.power_iv`` (checked against simulated rejection
rates), and ``sp.identify`` (estimands checked numerically against the
interventional distribution of random structural models).

Graph identities. The do-calculus rules on graphs where the answer follows
from d-separation by inspection.

Recoveries. One simulated design per estimator with a planted parameter,
four standard errors (or four across-seed standard deviations, quoted)
as the band.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import statspai as sp
from statspai.dag import identification as ident

pytestmark = pytest.mark.filterwarnings("ignore")


def _within(estimate: float, truth: float, se: float, n_sigma: float = 4.0) -> bool:
    return bool(np.isfinite(se) and se > 0 and abs(estimate - truth) <= n_sigma * se)


# --------------------------------------------------------------------- #
#  event_study: never-treated coding and empty event times
# --------------------------------------------------------------------- #


def _two_group_panel(first_period: int, never: float, n: int = 200) -> pd.DataFrame:
    """Half the units treated four periods in; effect 2.0; eight periods."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        treated = i < n // 2
        g = (first_period + 4) if treated else never
        for t in range(first_period, first_period + 8):
            post = treated and t >= first_period + 4
            y = 0.3 * t + (2.0 if post else 0.0) + (i % 5) + rng.normal()
            rows.append((i, t, g, y))
    return pd.DataFrame(rows, columns=["id", "t", "cohort", "y"])


class TestEventStudyNeverTreatedCoding:
    @pytest.mark.parametrize(
        "first_period, never",
        [(1, np.nan), (1, 0), (1, 9999), (1, np.inf), (2000, 0), (2000, np.nan)],
    )
    def test_every_documented_coding_gives_the_same_estimate(self, first_period, never):
        ref = sp.event_study(
            _two_group_panel(1, np.nan),
            y="y",
            treat_time="cohort",
            time="t",
            unit="id",
            window=(-4, 3),
        )
        res = sp.event_study(
            _two_group_panel(first_period, never),
            y="y",
            treat_time="cohort",
            time="t",
            unit="id",
            window=(-4, 3),
        )
        # 0 with periods 1..8 used to put the controls in the post bins.
        assert res.estimate == pytest.approx(ref.estimate, abs=1e-9)
        assert res.se == pytest.approx(ref.se, abs=1e-9)
        assert _within(res.estimate, 2.0, res.se)

    def test_treatment_time_equal_to_the_first_period_warns(self):
        df = _two_group_panel(0, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=sp.AssumptionWarning)
            with pytest.raises(sp.AssumptionWarning, match="first period"):
                sp.event_study(
                    df, y="y", treat_time="cohort", time="t", unit="id", window=(-4, 3)
                )

    def test_event_time_without_treated_rows_is_left_out(self):
        df = _two_group_panel(1, np.nan)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wide = sp.event_study(
                df, y="y", treat_time="cohort", time="t", unit="id", window=(-4, 4)
            )
        assert any("no treated observation" in str(w.message) for w in caught)
        exact = sp.event_study(
            df, y="y", treat_time="cohort", time="t", unit="id", window=(-4, 3)
        )
        # Exposure runs to +3. Asking for +4 used to add a row with
        # coefficient 0 and SE 0 and average it in: 1.56 for a 2.0 effect.
        assert wide.model_info["empty_event_times"] == [4]
        assert 4 not in wide.detail["relative_time"].tolist()
        assert wide.estimate == pytest.approx(exact.estimate, abs=1e-9)
        assert wide.se == pytest.approx(exact.se, abs=1e-9)


# --------------------------------------------------------------------- #
#  Power calculators against simulated rejection rates
# --------------------------------------------------------------------- #


class TestPowerAgainstSimulation:
    def test_power_did_under_ar1_errors(self):
        rng = np.random.default_rng(1)
        n, T, T_post, rho, es, reps = 200, 10, 5, 0.5, 0.2, 3000
        treated = np.arange(n) < n // 2
        reject = 0
        for _ in range(reps):
            e = np.zeros((n, T))
            e[:, 0] = rng.normal(size=n)
            for t in range(1, T):
                e[:, t] = rho * e[:, t - 1] + np.sqrt(1 - rho**2) * rng.normal(size=n)
            y = e + rng.normal(size=n)[:, None]  # plus a unit effect
            y[treated, T - T_post :] += es
            change = y[:, T - T_post :].mean(1) - y[:, : T - T_post].mean(1)
            est = change[treated].mean() - change[~treated].mean()
            se = np.sqrt(
                change[treated].var(ddof=1) / treated.sum()
                + change[~treated].var(ddof=1) / (~treated).sum()
            )
            reject += abs(est / se) > 1.96
        simulated = reject / reps
        formula = float(
            sp.power_did(
                n=n, effect_size=es, n_periods=T, n_treated_periods=T_post, rho=rho
            ).power
        )
        # MC SE of the simulated rate is 0.009. The old formula gave 0.48.
        assert formula == pytest.approx(simulated, abs=0.035)

    def test_power_did_unequal_arms(self):
        base = sp.power_did(n=400, effect_size=0.2, n_periods=6, n_treated_periods=3)
        skew = sp.power_did(
            n=400, effect_size=0.2, n_periods=6, n_treated_periods=3, prop_treat=0.2
        )
        # Var is proportional to 1 / (p (1 - p)): 4 at p = 0.5, 6.25 at 0.2.
        z = norm.ppf(0.975)
        ncp_base = norm.ppf(base.power) + z
        ncp_skew = norm.ppf(skew.power) + z
        assert ncp_skew / ncp_base == pytest.approx(np.sqrt(4 / 6.25), abs=1e-9)

    def test_power_rd_is_the_local_linear_test(self):
        rng = np.random.default_rng(0)
        n, es, h, reps = 2000, 0.25, 0.5, 2000
        reject = 0
        for _ in range(reps):
            x = rng.uniform(0, 1, n) - 0.5
            y = es * (x >= 0) + 0.3 * x + rng.normal(size=n)
            w = np.clip(1 - np.abs(x / h), 0, None)
            D = (x >= 0).astype(float)
            X = np.column_stack([np.ones(n), x, D, D * x])
            bread = np.linalg.inv(X.T @ (X * w[:, None]))
            beta = bread @ X.T @ (w * y)
            score = X * (w * (y - X @ beta))[:, None]
            V = bread @ (score.T @ score) @ bread
            reject += abs(beta[2] / np.sqrt(V[2, 2])) > 1.96
        simulated = reject / reps
        formula = float(sp.power_rd(n=n, effect_size=es, bandwidth=h).power)
        # MC SE 0.010. The old formula gave 0.93.
        assert formula == pytest.approx(simulated, abs=0.04)

    def test_power_iv_scales_with_the_first_stage(self):
        rng = np.random.default_rng(0)
        n, b, reps = 400, 0.2, 3000
        reject, f_stats = 0, []
        for _ in range(reps):
            z = rng.normal(size=n)
            d = 0.3 * z + rng.normal(size=n)
            y = b * d + rng.normal(size=n)
            dz = d @ z
            b_iv = (y @ z) / dz
            e = y - b_iv * d
            se = np.sqrt(e @ e / (n - 1) * (z @ z)) / abs(dz)
            reject += abs(b_iv / se) > 1.96
            pi = dz / (z @ z)
            r = d - pi * z
            f_stats.append(pi**2 * (z @ z) / (r @ r / (n - 1)))
        simulated = reject / reps
        formula = float(
            sp.power_iv(n=n, effect_size=b, first_stage_f=float(np.mean(f_stats))).power
        )
        # MC SE 0.007; sd(d) is 1.04, not 1, hence the slack. The old
        # formula gave 0.95.
        assert formula == pytest.approx(simulated, abs=0.04)

    def test_dispatcher_reaches_the_same_formulas(self):
        direct = sp.power_did(
            n=1000, effect_size=0.1, n_periods=10, n_treated_periods=5
        )
        routed = sp.power(
            "did", n=1000, effect_size=0.1, n_periods=10, n_treated_periods=5
        )
        assert float(routed.power) == pytest.approx(float(direct.power), abs=1e-12)


# --------------------------------------------------------------------- #
#  identify: estimands against random structural models
# --------------------------------------------------------------------- #


def _scm(dag, rng):
    nodes = sorted(dag._nodes)
    parents = {v: sorted(p for p, ch in dag._edges.items() if v in ch) for v in nodes}
    tables = {v: rng.uniform(0.1, 0.9, size=(2,) * len(parents[v])) for v in nodes}
    return nodes, parents, tables


def _joint(dag, scm, do=None):
    """Exact distribution of the observed variables, optionally under do()."""
    nodes, parents, tables = scm
    observed = [v for v in nodes if not ident._is_latent(v)]
    latent = [v for v in nodes if ident._is_latent(v)]
    order = latent + ident._topo_order(dag, observed)
    out: dict = {}
    for combo in itertools.product((0, 1), repeat=len(order)):
        val = dict(zip(order, combo))
        pr = 1.0
        for v in order:
            if do and v in do:
                pr *= 1.0 if val[v] == do[v] else 0.0
                continue
            p1 = tables[v][tuple(val[q] for q in parents[v])]
            pr *= p1 if val[v] == 1 else 1 - p1
        key = tuple(sorted((v, val[v]) for v in observed))
        out[key] = out.get(key, 0.0) + pr
    return out


_IDENTIFIABLE = [
    "Z -> X; Z -> Y; X -> Y",  # back-door
    "X -> M; M -> Y; X <-> Y",  # front-door
    "X -> M; M -> Y",
    "W -> X; X -> M; M -> Y; W -> Y",
    "X -> Z1; Z1 -> Z2; Z2 -> Y; X <-> Z2; Z1 <-> Y",
    "Z -> X; X -> M; M -> Y; X <-> Y; Z -> M",
    "X -> M1; M1 -> M2; M2 -> Y; X <-> Y; X <-> M2",
]


class TestIdentifyEstimands:
    @pytest.mark.parametrize("spec", _IDENTIFIABLE)
    def test_estimand_equals_the_interventional_distribution(self, spec):
        g = sp.dag(spec)
        res = sp.identify(g, treatment="X", outcome="Y")
        assert res.identifiable
        worst = 0.0
        for seed in range(4):
            scm = _scm(g, np.random.default_rng(seed))
            observational = _joint(g, scm)
            for x in (0, 1):
                interventional = _joint(g, scm, do={"X": x})
                for y in (0, 1):
                    truth = sum(
                        pr for k, pr in interventional.items() if dict(k)["Y"] == y
                    )
                    got = ident._evaluate(
                        res._expression, observational, {"X": x, "Y": y}
                    )
                    worst = max(worst, abs(got - truth))
        # Exact enumeration on both sides; observed 2e-16.
        assert worst < 1e-10

    def test_front_door_formula_as_printed(self):
        res = sp.identify(sp.dag("X -> M; M -> Y; X <-> Y"), treatment="X", outcome="Y")
        # The inner sum runs over a copy of the treatment, written X'. The
        # old output was sum_{M} [P(Y) * P(M | X)], which is P(Y).
        assert res.estimand == ("sum_{M} [P(M | X) * sum_{X'} [P(X') * P(Y | M, X')]]")

    def test_back_door_formula_as_printed(self):
        res = sp.identify(sp.dag("Z -> X; Z -> Y; X -> Y"), treatment="X", outcome="Y")
        assert res.estimand == "sum_{Z} [P(Y | X, Z) * P(Z)]"

    @pytest.mark.parametrize(
        "spec",
        ["X -> Y; X <-> Y", "Z -> X; X -> Y; X <-> Y", "X -> Z; Z -> Y; X <-> Z"],
    )
    def test_hedges_are_not_identifiable(self, spec):
        res = sp.identify(sp.dag(spec), treatment="X", outcome="Y")
        assert not res.identifiable
        assert res.hedge is not None


class TestDoCalculusRules:
    def test_rule1_drops_an_observation_that_is_d_separated(self):
        # Z -> X -> Y: with X set, Z says nothing about Y.
        assert sp.do_rule1(sp.dag("Z -> X -> Y"), Y="Y", X="X", Z="Z").applicable
        # A direct edge Z -> Y keeps Z informative.
        assert not sp.do_rule1(
            sp.dag("Z -> X -> Y; Z -> Y"), Y="Y", X="X", Z="Z"
        ).applicable

    def test_rule2_exchanges_action_and_observation_without_a_back_door(self):
        assert sp.do_rule2(sp.dag("X -> Y; Z -> Y"), Y="Y", X="X", Z="Z").applicable
        assert not sp.do_rule2(
            sp.dag("Z -> Y; Z <-> Y"), Y="Y", X=None, Z="Z"
        ).applicable

    def test_rule3_drops_an_action_with_no_path_to_the_outcome(self):
        assert sp.do_rule3(sp.dag("X -> Y; Z -> W"), Y="Y", X="X", Z="Z").applicable
        assert not sp.do_rule3(sp.dag("X -> Y; Z -> Y"), Y="Y", X="X", Z="Z").applicable

    def test_apply_reports_all_three_rules_in_order(self):
        checks = sp.do_calculus_apply(sp.dag("Z -> X -> Y"), Y="Y", X="X", Z="Z")
        assert [c.rule for c in checks] == [1, 2, 3]
        assert all(c.applicable for c in checks)

    def test_swig_splits_the_intervened_node(self):
        sw = sp.swig(sp.dag("L -> X; L -> Y; X -> Y"), {"X": "x"})
        assert sorted(sw.nodes) == ["L(X=x)", "X", "X(x)", "Y(X=x)"]
        assert "Y(X=x)" in sw.counterfactual_nodes()


# --------------------------------------------------------------------- #
#  Identities
# --------------------------------------------------------------------- #


def test_rd_distributional_design_level_jump_is_the_cdf_effect():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.uniform(-1, 1, n)
    y = 0.5 * x + rng.normal(0, 1, n) + 1.0 * (x >= 0)
    df = pd.DataFrame({"y": y, "x": x})
    q = np.array([0.25, 0.5, 0.75])
    ddd = sp.rd_distributional_design(df, y="y", running="x", quantiles=q)
    dist = sp.rd_distribution(df, y="y", running="x", quantiles=q)
    # Same local-linear regression of 1{Y <= y_q}; two entry points.
    np.testing.assert_allclose(ddd.rdd_effect, dist.cdf_effect, atol=1e-12)
    # A pure level shift moves mass up and leaves no kink.
    assert np.all(ddd.rdd_effect < 0)


def test_policy_weight_prte_is_a_rectangle_around_one_half():
    w = sp.policy_weight_prte(0.4)
    # Half-width |shift| / 2 = 0.2 around u = 0.5; points chosen off the
    # edges, where floating point decides.
    grid = np.array([0.1, 0.29, 0.31, 0.5, 0.69, 0.71, 0.9])
    assert w(grid).tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]


# --------------------------------------------------------------------- #
#  Recoveries
# --------------------------------------------------------------------- #


def _sharp_rd(seed: int, n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    y = 1.0 * (x >= 0) + 0.5 * x + 0.3 * z1 + rng.normal(0, 0.3, n)
    return pd.DataFrame({"x": x, "y": y, "z1": z1, "z2": z2})


def test_rdbalance_does_not_reject_balanced_covariates():
    rejections = []
    for s in range(40):
        table = sp.rdbalance(_sharp_rd(s), x="x", c=0, covs=["z1", "z2"])
        rejections.append((table["pvalue"] < 0.05).mean())
    # Two tests per sample, both nulls true: 80 tests at 5%.
    assert np.mean(rejections) <= 0.125


def test_rd_interference_recovers_direct_and_spillover_jumps():
    est = []
    for s in range(12):
        rng = np.random.default_rng(s)
        n = 3000
        x = rng.uniform(-1, 1, n)
        xn = rng.uniform(-1, 1, n)
        y = (
            1.0 * (x >= 0)
            + 0.4 * (xn >= 0)
            + 0.5 * x
            + 0.2 * xn
            + rng.normal(0, 0.3, n)
        )
        res = sp.rd_interference(
            pd.DataFrame({"y": y, "x": x, "xn": xn}),
            y="y",
            running="x",
            neighbour_running="xn",
        )
        est.append((res.direct_effect, res.spillover_effect))
    est = np.array(est)
    mc_se = est.std(axis=0, ddof=1) / np.sqrt(len(est))
    assert abs(est[:, 0].mean() - 1.0) <= 4 * mc_se[0]
    assert abs(est[:, 1].mean() - 0.4) <= 4 * mc_se[1]


def test_rd_multi_extrapolate_interpolates_between_cutoffs():
    draws = []
    for s in range(12):
        rng = np.random.default_rng(s)
        n = 6000
        x = rng.uniform(0, 3, n)
        # Effect 0.5 at the cutoff 1.0 and 1.0 at 2.0; linear in between.
        y = 0.3 * x + 0.5 * (x >= 1.0) + 1.0 * (x >= 2.0) + rng.normal(0, 0.3, n)
        res = sp.rd_multi_extrapolate(
            pd.DataFrame({"y": y, "x": x}),
            y="y",
            x="x",
            cutoffs=[1.0, 2.0],
            eval_points=np.array([1.5]),
        )
        draws.append((res.estimate, res.se))
    est, se = np.array(draws).T
    assert abs(est.mean() - 0.75) <= 4 * est.std(ddof=1) / np.sqrt(len(est))
    assert 0.6 <= se.mean() / est.std(ddof=1) <= 1.6


def _one_cohort_panel(seed: int, n_units: int = 150) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        g = [0, 5][i % 2]
        a = rng.normal()
        for t in range(1, 9):
            d = 1.0 if (g and t >= g) else 0.0
            rows.append((i, t, g, a + 0.3 * t + 2.0 * d + rng.normal(0, 0.5)))
    return pd.DataFrame(rows, columns=["id", "t", "cohort", "y"])


def test_bjs_pretrend_joint_has_the_right_size():
    reject = []
    for s in range(20):
        df = _one_cohort_panel(s)
        imp = sp.did_imputation(
            df,
            y="y",
            group="id",
            time="t",
            first_treat="cohort",
            horizon=[-3, -2, -1, 0, 1, 2],
        )
        out = sp.bjs_pretrend_joint(
            imp,
            df,
            y="y",
            group="id",
            time="t",
            first_treat="cohort",
            n_boot=80,
            seed=s,
        )
        reject.append(out["pvalue"] < 0.05)
    # No pre-trend in the DGP: at most 4 of 20 at a true 5% has
    # probability 0.984.
    assert sum(reject) <= 4


def test_did_bcf_recovers_the_att():
    res = sp.did_bcf(
        _one_cohort_panel(0, n_units=120),
        y="y",
        treat="cohort",
        time="t",
        id="id",
        n_bootstrap=60,
        seed=0,
    )
    assert _within(res.estimate, 2.0, res.se)


def test_pci_mtp_recovers_a_shift_effect():
    rng = np.random.default_rng(0)
    n = 4000
    U = rng.normal(size=n)
    A = 0.5 * U + rng.normal(size=n)
    Z = 0.7 * U + 0.5 * rng.normal(size=n)
    W = 0.7 * U + 0.5 * rng.normal(size=n)
    Y = 1.5 * A + U + rng.normal(size=n)
    res = sp.pci_mtp(
        pd.DataFrame({"Y": Y, "A": A, "Z": Z, "W": W}),
        y="Y",
        treat="A",
        proxy_z=["Z"],
        proxy_w=["W"],
        delta=1.0,
        n_boot=60,
        seed=0,
    )
    # Shifting the treatment by one unit moves the outcome by 1.5. The
    # confounded regression slope is 1.9.
    assert _within(res.estimate, 1.5, res.se)


def test_select_pci_proxies_ranks_the_true_proxies_first():
    rng = np.random.default_rng(0)
    n = 4000
    u = rng.normal(size=n)
    z = u + rng.normal(scale=0.5, size=n)
    w = u + rng.normal(scale=0.5, size=n)
    d = (z + rng.normal(size=n) > 0).astype(float)
    y = 1.0 + 0.5 * d + w + rng.normal(size=n)
    df = pd.DataFrame(
        {
            "y": y,
            "d": d,
            "z": z,
            "w": w,
            "j1": rng.normal(size=n),
            "j2": rng.normal(size=n),
        }
    )
    res = sp.select_pci_proxies(df, y="y", treat="d", candidates=["z", "w", "j1", "j2"])
    assert res.recommended_z[0] == "z"
    assert res.recommended_w[0] == "w"


def test_ml_bounds_contain_the_effect_and_keep_the_manski_width():
    rng = np.random.default_rng(0)
    n = 3000
    x1 = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-x1)))
    y = np.clip(0.3 + 0.2 * d + 0.1 * x1 + rng.normal(0, 0.1, n), 0, 1)
    res = sp.ml_bounds(
        pd.DataFrame({"y": y, "d": d, "x1": x1}),
        y="y",
        treat="d",
        covariates=["x1"],
        y_min=0,
        y_max=1,
        n_bootstrap=20,
        random_state=0,
    )
    assert res.lower <= 0.2 <= res.upper
    # Worst-case bounds on an outcome in [0, 1] have width one.
    assert res.upper - res.lower == pytest.approx(1.0, abs=0.02)


def test_honest_forests_recover_the_regression_function():
    rng = np.random.default_rng(0)
    n = 3000
    X = rng.normal(size=(n, 3))
    truth = np.sin(X[:, 0]) + 0.5 * X[:, 1]
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["y"] = truth + rng.normal(0, 0.3, n)
    fit = sp.regression_forest(
        df, y="y", covariates=["a", "b", "c"], n_estimators=300, random_state=0
    )
    pred = np.asarray(fit.predictions, dtype=float).ravel()
    assert 1 - np.nanmean((pred - truth) ** 2) / truth.var() > 0.95

    p = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    df["label"] = rng.binomial(1, p)
    prob = sp.probability_forest(
        df, y="label", covariates=["a", "b", "c"], n_estimators=300, random_state=0
    )
    probs = np.asarray(prob.predictions, dtype=float)
    np.testing.assert_allclose(np.nansum(probs, axis=1), 1.0, atol=1e-9)
    assert np.sqrt(np.nanmean((probs[:, -1] - p) ** 2)) < 0.1

    Y = np.c_[X[:, 0], -X[:, 0] + 0.5 * X[:, 1]] + rng.normal(0, 0.3, size=(n, 2))
    multi = sp.multi_regression_forest(
        y=Y, covariates=X, n_estimators=300, random_state=0
    )
    P = np.asarray(multi.predictions, dtype=float)
    target = np.c_[X[:, 0], -X[:, 0] + 0.5 * X[:, 1]]
    for j in (0, 1):
        assert 1 - np.nanmean((P[:, j] - target[:, j]) ** 2) / target[:, j].var() > 0.95


def test_instrumental_forest_recovers_the_complier_effect_function():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=(n, 3))
    z = rng.binomial(1, 0.5, n)
    u = rng.normal(size=n)
    w = (1.2 * z + 0.6 * u + rng.normal(scale=0.5, size=n) > 0.6).astype(float)
    y = (1 + x[:, 0]) * w + u + rng.normal(size=n)
    df = pd.DataFrame(
        {"y": y, "w": w, "z": z, "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2]}
    )
    fit = sp.instrumental_forest(
        df,
        y="y",
        treat="w",
        instrument="z",
        covariates=["x1", "x2", "x3"],
        n_estimators=400,
        random_state=0,
    )
    cate = np.asarray(fit.cate, dtype=float).ravel()
    ok = np.isfinite(cate)
    # Effect 1 + x1; across six seeds the mean is 1.06 and the slope 0.91
    # (a forest shrinks towards the mean).
    assert cate[ok].mean() == pytest.approx(1.0, abs=0.25)
    assert np.polyfit(x[ok, 0], cate[ok], 1)[0] == pytest.approx(1.0, abs=0.3)


def test_causal_survival_recovers_the_rmst_difference():
    rng_truth = np.random.default_rng(99)
    N = 400_000
    xt = rng_truth.normal(size=N)
    t1 = rng_truth.exponential(1 / np.exp(0.5 * xt - 0.5))
    t0 = rng_truth.exponential(1 / np.exp(0.5 * xt))
    truth = float(np.mean(np.minimum(t1, 1.0) - np.minimum(t0, 1.0)))

    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=(n, 2))
    w = rng.binomial(1, 0.5, n)
    T = rng.exponential(1 / np.exp(0.5 * x[:, 0] - 0.5 * w))
    C = rng.exponential(3, n)
    df = pd.DataFrame(
        {
            "t": np.minimum(T, C),
            "d": (T <= C).astype(int),
            "w": w,
            "x1": x[:, 0],
            "x2": x[:, 1],
        }
    )
    res = sp.causal_survival(
        df,
        time="t",
        event="d",
        treat="w",
        covariates=["x1", "x2"],
        horizon=1.0,
        n_estimators=300,
        random_state=0,
    )
    assert truth == pytest.approx(0.115, abs=0.005)
    assert _within(res.ate, truth, res.se)


def test_nonlinear_icp_never_returns_the_child():
    for s in range(10):
        rng = np.random.default_rng(s)
        n = 800
        env = np.r_[np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)]
        x1 = 2.0 * env + rng.normal(size=n)
        y = np.sin(x1) + 0.5 * x1 + 0.3 * rng.normal(size=n)
        x2 = y + rng.normal(size=n)
        res = sp.nonlinear_icp(pd.DataFrame({"X1": x1, "X2": x2}), y, env, alpha=0.05)
        assert "X2" not in res.parents


def test_counterfactual_policy_value_under_a_linear_scm():
    rng = np.random.default_rng(0)
    n = 5000
    s = rng.normal(0, 1, n)
    a = 0.5 * s + rng.normal(0, 1, n)
    r = 1.0 * s + 2.0 * a + rng.normal(0, 1, n)
    res = sp.counterfactual_policy_optimization(
        pd.DataFrame({"s": s, "a": a, "r": r}),
        state="s",
        action="a",
        reward="r",
        target_policy=lambda si: si + 1.0,
    )
    # Under a = s + 1: E[r] = E[s] + 2 E[s + 1] = 2; logged E[r] = 0.
    assert res.expected_value_target == pytest.approx(2.0, abs=0.1)
    assert res.expected_value_logged == pytest.approx(0.0, abs=0.15)
    assert res.improvement == pytest.approx(
        res.expected_value_target - res.expected_value_logged, abs=1e-12
    )


def test_structural_mdp_recovers_transition_and_reward_coefficients():
    rng = np.random.default_rng(0)
    n = 4000
    s1, s2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    a1 = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "s1": s1,
            "s2": s2,
            "a1": a1,
            "ns1": 0.8 * s1 + 0.2 * a1 + rng.normal(0, 0.1, n),
            "ns2": 0.5 * s2 + 0.3 * a1 + rng.normal(0, 0.1, n),
            "r": 1.0 * s1 + 0.5 * a1 + rng.normal(0, 0.1, n),
        }
    )
    res = sp.structural_mdp(
        df,
        state_cols=["s1", "s2"],
        action_cols=["a1"],
        reward="r",
        next_state_cols=["ns1", "ns2"],
    )
    np.testing.assert_allclose(res.A, [[0.8, 0.0], [0.0, 0.5]], atol=0.02)
    np.testing.assert_allclose(np.ravel(res.B), [0.2, 0.3], atol=0.02)
    np.testing.assert_allclose(np.ravel(res.reward_coef), [1.0, 0.0, 0.5], atol=0.02)


def test_llm_annotator_correct_undoes_misclassification_attenuation():
    corrected, naive = [], []
    for s in range(15):
        rng = np.random.default_rng(s)
        n, n_val = 3000, 300
        T = (rng.random(n) > 0.5).astype(int)
        noisy = (T ^ (rng.random(n) < 0.15)).astype(int)  # 15% flipped
        y = 1.0 * T + rng.standard_normal(n)
        human = pd.Series([T[i] if i < n_val else np.nan for i in range(n)])
        res = sp.llm_annotator_correct(
            annotations_llm=pd.Series(noisy),
            outcome=pd.Series(y),
            annotations_human=human,
        )
        corrected.append(res.estimate)
        naive.append(res.naive_estimate)
    corrected = np.array(corrected)
    # True effect 1.0; the uncorrected contrast is attenuated to 0.70.
    assert abs(corrected.mean() - 1.0) <= 4 * corrected.std(ddof=1) / np.sqrt(15)
    assert np.mean(naive) == pytest.approx(0.70, abs=0.05)
