"""Defects found in the October 2026 review of the design-side modules
(``doe``, ``matching``, ``experimental``, ``interference``, ``ope``).

Each test asserts a known truth, an invariance, equality with the
hand-coded call, or a specific exception or warning class.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    ConvergenceWarning,
    IdentificationFailure,
    MethodIncompatibility,
)

# ---------------------------------------------------------------------
# 1. factor_importance with tied numeric factors
# ---------------------------------------------------------------------


def _two_cubed() -> pd.DataFrame:
    d = sp.factorial_design(["A", "B", "C"], replicates=25).design[["A", "B", "C"]]
    d = d.reset_index(drop=True)
    rng = np.random.default_rng(0)
    d["y"] = 2 * d.A + 1 * d.B + 0.5 * rng.normal(size=len(d))
    return d


def test_factor_importance_numeric_coding_equals_labels():
    d = _two_cubed()
    num = sp.factor_importance(d, "y").importance
    lab = sp.factor_importance(d.astype({"A": str, "B": str, "C": str}), "y").importance
    # same cells, same within-cell variances
    np.testing.assert_allclose(num.to_numpy(), lab.to_numpy(), rtol=0, atol=1e-12)


def test_factor_importance_does_not_depend_on_row_order():
    d = _two_cubed()
    a = sp.factor_importance(d, "y").importance
    b = sp.factor_importance(d.sample(frac=1, random_state=0), "y").importance
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-12)


def test_factor_importance_known_shares_on_two_cubed():
    # y = 2A + B + noise(sd 0.5) on +-1 factors: explained variance 4 + 1,
    # so dropping A loses 4 / 5 of it, dropping B 1 / 5, and C nothing.
    imp = sp.factor_importance(_two_cubed(), "y").importance
    assert imp["C"] == 0.0
    assert abs(imp["A"] - 0.8) < 0.03
    assert abs(imp["B"] - 0.2) < 0.03


def test_factor_importance_integer_factor_next_to_continuous():
    rng = np.random.default_rng(3)
    n = 400
    x, g = rng.normal(size=n), rng.integers(1, 4, n)
    d = pd.DataFrame({"y": x + 2 * (g == 2) + 0.3 * rng.normal(size=n), "x": x, "g": g})
    num = sp.factor_importance(d, "y").importance
    lab = sp.factor_importance(d.assign(g=d.g.astype(str)), "y").importance
    np.testing.assert_allclose(num.to_numpy(), lab.to_numpy(), rtol=0, atol=1e-12)
    assert 0 <= num.max() <= 1  # a share of explained variance


def test_factor_importance_refuses_outcome_among_factors():
    with pytest.raises(MethodIncompatibility, match="among the factors"):
        sp.factor_importance(_two_cubed(), "y", factors=["A", "y"])


# ---------------------------------------------------------------------
# 2. mdp_policy_value: positivity and labelling of the states
# ---------------------------------------------------------------------

_P0 = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6]])
_P1 = np.array([[0.2, 0.3, 0.5], [0.1, 0.3, 0.6], [0.05, 0.25, 0.7]])


def _mdp(p_treat, n=1500, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    s, rows = 0, []
    for _ in range(n):
        w = int(rng.random() < p_treat[s])
        rows.append((s, w, s + w * (1 + s) + rng.normal()))
        s = int(rng.choice(3, p=(_P1 if w else _P0)[s]))
    return pd.DataFrame(rows, columns=["s", "w", "y"])


def test_mdp_policy_value_refuses_policy_without_support():
    df = _mdp([0.3, 0.5, 1.0])  # action 0 never taken in state 2
    with pytest.raises(IdentificationFailure, match="state.*'2'"):
        sp.mdp_policy_value(df, "y", "w", ["s"], policy=0)
    with pytest.raises(IdentificationFailure):
        sp.mdp_policy_value(df, "y", "w", ["s"], policy=1, baseline=0)
    # the policy that the data do support is still evaluated
    assert np.isfinite(sp.mdp_policy_value(df, "y", "w", ["s"], policy=1).value)


def test_mdp_policy_value_does_not_depend_on_state_labels():
    df = _mdp([0.3, 0.5, 0.7])
    base = sp.mdp_policy_value(df, "y", "w", ["s"], policy=1)
    for relabel in ({0: 2, 1: 1, 2: 0}, {0: "c", 1: "a", 2: "b"}, {0: 7, 1: 3, 2: 5}):
        other = sp.mdp_policy_value(
            df.assign(s=df.s.map(relabel)), "y", "w", ["s"], policy=1
        )
        assert other.value == pytest.approx(base.value, rel=1e-12)
        assert other.se == pytest.approx(base.se, rel=1e-10)
        assert other.detail["min_omega"] == pytest.approx(
            base.detail["min_omega"], rel=1e-10
        )


def test_mdp_linear_basis_depends_only_on_the_span_of_the_features():
    # [1, s] spans the same functions as [1, a + b s]: value and standard
    # error must not move with the origin or the unit of the feature
    rng = np.random.default_rng(0)
    s, rows = 0.0, []
    for _ in range(800):
        w = int(rng.random() < 0.5)
        rows.append((s, w, s + w + rng.normal()))
        s = 0.6 * s + 0.5 * w + 0.5 * rng.normal()
    df = pd.DataFrame(rows, columns=["s", "w", "y"])
    kw = dict(policy=1, baseline=0, propensity=0.5, features=["s"])
    base = sp.mdp_policy_value(df, "y", "w", ["s"], **kw)
    for shift, scale in ((10.0, 1.0), (1000.0, 1.0), (0.0, 250.0), (-3.0, 0.01)):
        other = sp.mdp_policy_value(
            df.assign(s=shift + scale * df.s), "y", "w", ["s"], **kw
        )
        assert other.value == pytest.approx(base.value, rel=1e-8)
        assert other.se == pytest.approx(base.se, rel=1e-8)
        assert other.detail["min_omega"] == pytest.approx(
            base.detail["min_omega"], rel=1e-6, abs=1e-8
        )


def test_mdp_tabular_value_is_the_plug_in_of_the_fitted_transition_model():
    df = _mdp([0.3, 0.5, 0.7], n=1200)
    res = sp.mdp_policy_value(df, "y", "w", ["s"], policy=1)
    # stationary mean reward of the transition model fitted to the periods
    # in which the action taken was the policy's
    cur, nxt = df.iloc[:-1], df["s"].to_numpy()[1:]
    on = cur["w"].to_numpy() == 1
    P = np.zeros((3, 3))
    np.add.at(P, (cur["s"].to_numpy()[on], nxt[on]), 1.0)
    P /= P.sum(axis=1, keepdims=True)
    r = cur[on].groupby("s")["y"].mean().to_numpy()
    vals, vecs = np.linalg.eig(P.T)
    pi = np.real(vecs[:, np.argmax(np.real(vals))])
    pi /= pi.sum()
    assert res.value == pytest.approx(float(pi @ r), rel=1e-9)
    assert res.value == pytest.approx(res.detail["bellman_value"], rel=1e-10)


def test_mdp_refuses_missing_period_inside_trajectory():
    df = _mdp([0.3, 0.5, 0.7], n=300)
    hole = df.copy()
    hole.loc[100, "y"] = np.nan
    with pytest.raises(MethodIncompatibility, match="inside a trajectory"):
        sp.mdp_policy_value(hole, "y", "w", ["s"], policy=1)
    with pytest.raises(MethodIncompatibility, match="inside a trajectory"):
        sp.marginal_policy_effect(hole, "y", "w", [], horizon=1, propensity=0.5)
    # a missing first period only shortens the trajectory
    edge = df.copy()
    edge.loc[0, "y"] = np.nan
    a = sp.mdp_policy_value(edge, "y", "w", ["s"], policy=1)
    b = sp.mdp_policy_value(df.iloc[1:], "y", "w", ["s"], policy=1)
    assert a.value == b.value and a.se == b.se


def test_marginal_policy_effect_hac_lags_bounds():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=200), "w": rng.binomial(1, 0.3, 200)})
    kw = dict(horizon=2, propensity=0.3)
    with pytest.raises(MethodIncompatibility, match="at least horizon"):
        sp.marginal_policy_effect(df, "y", "w", [], hac_lags=1, **kw)
    with pytest.raises(MethodIncompatibility, match="smaller than the number"):
        sp.marginal_policy_effect(df, "y", "w", [], hac_lags=5000, **kw)
    assert sp.marginal_policy_effect(df, "y", "w", [], hac_lags=2, **kw).se > 0


# ---------------------------------------------------------------------
# 3. categorical covariates
# ---------------------------------------------------------------------


def _cat_frame():
    rng = np.random.default_rng(0)
    n = 400
    x, g = rng.normal(size=n), rng.integers(1, 4, n)
    d = rng.binomial(1, 1 / (1 + np.exp(-(x + 1.0 * (g == 2)))))
    cat = pd.DataFrame(
        {
            "d": d,
            "x": x,
            "g": pd.Categorical(g),
            "y": 1 + d + x + 2 * (g == 2) + rng.normal(size=n),
        }
    )
    dummy = cat.assign(g2=(g == 2) * 1.0, g3=(g == 3) * 1.0)
    return cat, dummy


@pytest.mark.parametrize("as_text", [False, True])
def test_category_covariate_equals_dummy_coding(as_text):
    cat, dummy = _cat_frame()
    if as_text:
        cat = cat.assign(g=cat.g.astype(str))
    hand = ["x", "g2", "g3"]

    a, b = sp.pscore(cat, "d", ["x", "g"]), sp.pscore(dummy, "d", hand)
    np.testing.assert_array_equal(a.pscore.to_numpy(), b.pscore.to_numpy())
    np.testing.assert_array_equal(a.coefficients.to_numpy(), b.coefficients.to_numpy())
    assert list(a.coefficients.index) == ["x", "g[T.2]", "g[T.3]", "_cons"]

    a = sp.marginal_policy_effect(cat, "y", "d", ["x", "g"], horizon=0)
    b = sp.marginal_policy_effect(dummy, "y", "d", hand, horizon=0)
    assert (a.estimate, a.se) == (b.estimate, b.se)

    for inter in (False, True):
        wa = sp.implied_weights(cat, "d", ["x", "g"], interactions=inter)
        wb = sp.implied_weights(dummy, "d", hand, interactions=inter)
        np.testing.assert_array_equal(wa.to_numpy(), wb.to_numpy())


def test_distance_functions_refuse_categorical_covariates():
    cat, _ = _cat_frame()
    for frame in (cat, cat.assign(g=cat.g.astype(str))):
        with pytest.raises(MethodIncompatibility, match="indicator"):
            sp.energy_distance(frame, "d", ["x", "g"])
        with pytest.raises(MethodIncompatibility, match="indicator"):
            sp.balance_vs_randomization(frame, "d", ["x", "g"], n_sim=10)
        with pytest.raises(MethodIncompatibility, match="indicator"):
            sp.contextual_bandit(
                lambda k, x, rng: rng.normal(), frame[["x", "g"]], n_arms=2, sigma=1.0
            )


# ---------------------------------------------------------------------
# 4. interference_test: boolean focal mask
# ---------------------------------------------------------------------


def test_interference_test_boolean_mask_equals_indices():
    rng = np.random.default_rng(0)
    n = 120
    A = np.triu((rng.random((n, n)) < 0.04).astype(int), 1)
    A = A + A.T
    Z = (rng.random(n) < 0.5).astype(int)
    Y = 1.0 * Z + 2.0 * (A @ Z > 0) + rng.normal(size=n)
    kw = dict(n_perm=99, seed=1)
    mask = sp.interference_test(Y, Z, A, focal=np.arange(n) < 60, **kw)
    index = sp.interference_test(Y, Z, A, focal=np.arange(60), **kw)
    assert mask.n_focal == index.n_focal == 60
    assert mask.pvalue == index.pvalue and mask.statistic == index.statistic
    with pytest.raises(MethodIncompatibility, match="one entry per unit"):
        sp.interference_test(Y, Z, A, focal=np.ones(n - 1, dtype=bool), **kw)


# ---------------------------------------------------------------------
# 5. complete separation in the propensity model
# ---------------------------------------------------------------------


def _separated() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "d": (x > 0).astype(int),
            "x": x,
            "z": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )


def test_pscore_warns_on_separation_and_does_not_claim_balance():
    s = _separated()
    with pytest.warns(ConvergenceWarning, match="separation"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AssumptionWarning)
            res = sp.pscore(s, "d", ["x", "z"])
    assert res.separated and not res.balanced and res.n_tested_blocks == 0
    # no unit in a block that does not exist; a score of one is in the top block
    assert set(res.blocks["block"]) <= set(range(1, res.n_blocks + 1))
    assert res.block.min() >= 1
    assert int(res.blocks[["n_control", "n_treated"]].to_numpy().sum()) == len(s)
    with pytest.warns(AssumptionWarning, match="no block holds both"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            sp.pscore(s, "d", ["x", "z"])


def test_full_match_warns_on_separation():
    with pytest.warns(ConvergenceWarning, match="separation"):
        sp.full_match(_separated(), "y", "d", ["x", "z"])


def test_overlapping_scores_raise_no_separation_warning():
    cat, dummy = _cat_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sp.pscore(dummy, "d", ["x", "g2", "g3"])
    assert not res.separated and res.n_tested_blocks > 0


# ---------------------------------------------------------------------
# 6. implied_weights when the treatment is collinear with the covariates
# ---------------------------------------------------------------------


def test_implied_weights_refuses_perfectly_predicted_treatment():
    rng = np.random.default_rng(0)
    n = 200
    d = rng.binomial(1, 0.5, n)
    df = pd.DataFrame({"d": d, "x": rng.normal(size=n), "k": d * 1.0})
    with pytest.raises(MethodIncompatibility, match="predict the treatment"):
        sp.implied_weights(df, "d", ["x", "k"])


# ---------------------------------------------------------------------
# 7. split_data on data with few distinct rows
# ---------------------------------------------------------------------


def test_split_data_on_discrete_data():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.integers(0, 3, 200), "b": rng.integers(0, 4, 200)})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        auto = sp.split_data(df, seed=1)
        twin = sp.split_data(df, seed=1, method="twinning")
    assert auto.model_info["method"] == "twinning"
    assert len(auto.test) == 40 and len(auto.train) == 160
    # the fallback is the twinning split for the same seed
    np.testing.assert_array_equal(auto.test_index, twin.test_index)
    with pytest.raises(MethodIncompatibility, match="distinct rows"):
        sp.split_data(df, seed=1, method="support")
    with pytest.raises(MethodIncompatibility, match="whole number"):
        sp.split_data(df, seed=1, test_size=0.3)


# ---------------------------------------------------------------------
# 8. bandit allocation: floor, sigma
# ---------------------------------------------------------------------


def test_prob_floor_holds_during_forced_pulls():
    h = pd.DataFrame({"arm": list("aabb"), "y": [1.0, 1, 0, 0]})
    p = sp.bandit_allocate(h, "y", "arm", arms=list("abc"), sigma=1.0, prob_floor=0.1)
    assert p.to_dict() == pytest.approx({"a": 0.1, "b": 0.1, "c": 0.8})
    # without a floor the forced pull is deterministic, and reported so
    p0 = sp.bandit_allocate(h, "y", "arm", arms=list("abc"), sigma=1.0)
    assert p0.to_dict() == {"a": 0.0, "b": 0.0, "c": 1.0}
    exp = sp.bandit_experiment(
        lambda k, rng: rng.normal(), 40, n_arms=3, sigma=1.0, prob_floor=0.05, seed=0
    )
    probs = exp.data[["prob_0", "prob_1", "prob_2"]].to_numpy()
    assert probs.min() >= 0.05 - 1e-12
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-12)


def test_bandit_allocate_warns_when_sigma_cannot_be_estimated():
    h = pd.DataFrame({"arm": list("aabb"), "y": [1.0, 1, 0, 0]})
    with pytest.warns(AssumptionWarning, match="Pass sigma="):
        sp.bandit_allocate(h, "y", "arm")
    # with sigma given the allocation is the same in any unit of the reward
    a = sp.bandit_allocate(h, "y", "arm", sigma=1.0)
    b = sp.bandit_allocate(h.assign(y=h.y * 1000), "y", "arm", sigma=1000.0)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-12)


@pytest.mark.parametrize(
    "kw", [{"sigma": -1.0}, {"sigma": 0.0}, {"prob_floor": -0.2}, {"prob_floor": 0.6}]
)
def test_bandit_allocate_validates_sigma_and_floor(kw):
    h = pd.DataFrame({"arm": list("aabb"), "y": [1.0, 0.5, 0, 0.2]})
    with pytest.raises(MethodIncompatibility, match="sigma|prob_floor"):
        sp.bandit_allocate(h, "y", "arm", **{"sigma": 1.0, **kw})


def test_bandit_and_adaptive_inference_input_checks():
    h = pd.DataFrame({"arm": ["a", "a", None, "b"], "y": [1.0, 0.5, 0, 0.2]})
    with pytest.raises(MethodIncompatibility, match="missing values"):
        sp.bandit_allocate(h, "y", "arm", sigma=1.0)
    rng = np.random.default_rng(0)
    d = pd.DataFrame(
        {
            "y": rng.normal(size=60),
            "arm": np.tile([0, 1, 2], 20),
            "p0": 0.9,
            "p1": 0.3,
            "p2": 0.5,
            "q0": 0.2,
            "q1": 0.3,
            "q2": 0.5,
            "p": 0.25,
        }
    )
    with pytest.raises(MethodIncompatibility, match="sum to one"):
        sp.adaptive_inference(d, "y", "arm", probs=["p0", "p1", "p2"], method="aipw")
    with pytest.raises(MethodIncompatibility, match="disagrees"):
        sp.adaptive_inference(d, "y", "arm", prob="p", probs=["q0", "q1", "q2"])
    assert sp.adaptive_inference(d, "y", "arm", probs=["q0", "q1", "q2"]).n_obs == 60
    with pytest.raises(MethodIncompatibility, match="missing values"):
        sp.adaptive_inference(
            d.assign(arm=d.arm.where(d.index != 3)), "y", "arm", prob="p"
        )
    with pytest.raises(MethodIncompatibility, match="overwritten"):
        sp.contextual_bandit(
            lambda k, x, g: g.normal(),
            pd.DataFrame({"arm": rng.normal(size=30)}),
            n_arms=2,
            sigma=1.0,
            seed=0,
        )


# ---------------------------------------------------------------------
# 9. space_filling with qualitative factors
# ---------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_qualitative_levels_balanced_to_within_one(seed):
    quali = {"g": ["a", "b", "c"], "h": ["u", "v", "w"]}
    d = sp.space_filling(
        10, 2, qualitative=quali, seed=seed, iterations=200, n_starts=1
    ).design
    for name in quali:
        assert sorted(d[name].value_counts().tolist()) == [3, 3, 4]


def test_qualitative_full_replicates_hold_every_combination_equally():
    d = sp.space_filling(
        18,
        1,
        qualitative={"g": ["a", "b", "c"], "h": ["u", "v", "w"]},
        seed=2,
        iterations=200,
        n_starts=1,
    ).design
    assert d.groupby(["g", "h"]).size().tolist() == [2] * 9


# ---------------------------------------------------------------------
# 10. smaller input checks
# ---------------------------------------------------------------------


def test_alpha_is_validated():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {"x": x, "d": rng.binomial(1, 0.5, n), "y": x + rng.normal(size=n)}
    )
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.full_match(df, "y", "d", ["x"], alpha=2)
    plan = sp.switchback_design(40, seed=0).assign(y=rng.normal(size=40))
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.switchback(plan, "y", "treat", alpha=2, n_draws=0)


def test_doe_candidate_checks():
    with pytest.raises(MethodIncompatibility, match="missing or infinite"):
        sp.doe_optimal("x", candidates=pd.DataFrame({"x": [0, 1, np.nan]}))
    base = sp.space_filling(8, 2, seed=1, iterations=100, n_starts=1).design
    with pytest.raises(MethodIncompatibility, match="outside the region"):
        sp.design_augment(base, 1, candidates=pd.DataFrame({"x1": [5.0], "x2": [7.0]}))
    inside = pd.DataFrame({"x1": [0.5, 0.1], "x2": [0.5, 0.9]})
    assert sp.design_augment(base, 1, candidates=inside).design.shape == (9, 2)


def test_support_points_warns_when_not_converged():
    X = pd.DataFrame(np.random.default_rng(0).normal(size=(200, 2)))
    with pytest.warns(ConvergenceWarning, match="not converged"):
        res = sp.support_points(X, 10, seed=1, max_iter=2)
    assert res.model_info["converged"] is False


def test_ess_refuses_undefined_inputs():
    with pytest.raises(MethodIncompatibility, match="by contains missing"):
        sp.ess([1, 2, 3, 4], by=[0, 0, np.nan, 1])
    with pytest.warns(AssumptionWarning, match="every weight is zero"):
        assert np.isnan(sp.ess(np.zeros(4)))
    with pytest.warns(AssumptionWarning, match="every weight is zero in group"):
        out = sp.ess([0, 0, 1, 3], by=[0, 0, 1, 1])
    assert np.isnan(out[0]) and out[1] == pytest.approx(1.6)


def test_two_criteria_match_refuses_negative_caliper():
    cat, _ = _cat_frame()
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        sp.two_criteria_match(
            cat, "d", pair=[{"type": "caliper", "on": "x", "width": -3}]
        )


def test_alias_table_lists_effects_only():
    res = sp.factorial_design(5, n_runs=8, alias_order=3)
    words = {w.lstrip("-") for w in res.defining_relation}
    assert words  # a 2^(5-2) fraction has three defining words
    assert not words & set(res.aliases)
    assert all(m.lstrip("-") != "" for mates in res.aliases.values() for m in mates)
    # every listed alias really is the same column up to sign
    d = res.design
    for effect, mates in res.aliases.items():
        col = np.prod(d[list(effect)].to_numpy(), axis=1)
        for m in mates:
            sign = -1 if m.startswith("-") else 1
            other = np.prod(d[list(m.lstrip("-"))].to_numpy(), axis=1)
            np.testing.assert_array_equal(col, sign * other)


def test_factorial_effects_degenerate_outcomes():
    fd = sp.factorial_design(["A", "B", "C"]).design
    with pytest.raises(MethodIncompatibility, match="does not vary"):
        sp.factorial_effects(fd.assign(y=3.0), "y")
    # one exactly active effect: the others are zero up to rounding, the
    # pseudo standard error is not positive, and no p-value is invented
    res = sp.factorial_effects(fd.assign(y=1.0 + 0.3 * fd.A), "y")
    assert res.effects.loc["A", "effect"] == pytest.approx(0.6)
    assert res.effects["lenth_p"].isna().all()


def test_sequential_design_says_why_a_missing_output_is_refused():
    with pytest.raises(MethodIncompatibility, match="surrogate"):
        sp.sequential_design(
            lambda X: np.where(X["x"] > 0.5, np.nan, X["x"]),
            {"x": (0, 1)},
            n_new=1,
            n_init=6,
            seed=1,
        )


def test_weights_series_are_matched_on_the_index():
    rng = np.random.default_rng(0)
    n = 80
    x = rng.normal(size=n)
    df = pd.DataFrame({"d": rng.binomial(1, 0.5, n), "x": x, "z": rng.normal(size=n)})
    w = pd.Series(rng.uniform(0.5, 2, n), index=df.index)
    ref = sp.energy_distance(df, "d", ["x", "z"], weights=w.to_numpy())
    shuffled = w.sample(frac=1, random_state=1)
    assert sp.energy_distance(df, "d", ["x", "z"], weights=shuffled) == ref
    with pytest.raises(MethodIncompatibility, match="index"):
        sp.energy_distance(df, "d", ["x", "z"], weights=w.set_axis(np.arange(n) + 1000))

    ps = pd.Series(rng.uniform(0.2, 0.8, n), index=df.index)
    ref_w = sp.ps_weights(ps, df["d"])
    again = sp.ps_weights(ps, df["d"].sample(frac=1, random_state=2))
    pd.testing.assert_series_equal(ref_w, again)
    with pytest.raises(MethodIncompatibility, match="index"):
        sp.ps_weights(ps, df["d"].set_axis(np.arange(n) + 1000))
