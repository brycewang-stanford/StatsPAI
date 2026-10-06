"""``statspai.doe``: known-truth and boundary tests.

Comparisons with R are in ``reference_parity/test_joseph_doe_parity.py``.
Here every expected value is derived independently of the implementation:
closed-form optimal designs, textbook counts, functions with analytic
sensitivity indices, and invariants.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from numpy.polynomial import legendre
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

FAST = {"iterations": 3000, "n_starts": 1}


# ---------------------------------------------------------------------------
# Space-filling designs
# ---------------------------------------------------------------------------


class TestSpaceFilling:
    @pytest.mark.parametrize("method", ["maxpro", "maximin", "uniform", "lhs"])
    def test_latin_hypercube_property(self, method):
        """One run in each of n equal slices of every factor."""
        d = sp.space_filling(12, 3, method=method, seed=1, polish=False, **FAST)
        for col in d.unit.T:
            assert sorted(np.floor(col * 12).astype(int)) == list(range(12))

    def test_bounds_are_respected_and_named(self):
        d = sp.space_filling(10, {"beta": (0.9, 0.99), "sigma": (1, 5)}, seed=2, **FAST)
        assert list(d.design.columns) == ["beta", "sigma"]
        assert d.design["beta"].between(0.9, 0.99).all()
        assert d.design["sigma"].between(1, 5).all()

    def test_search_improves_on_a_random_latin_hypercube(self):
        lhs = [sp.space_filling(15, 3, method="lhs", seed=s).criteria for s in range(8)]
        mp = sp.space_filling(15, 3, method="maxpro", seed=0, **FAST).criteria
        mm = sp.space_filling(15, 3, method="maximin", seed=0, **FAST).criteria
        un = sp.space_filling(15, 3, method="uniform", seed=0, **FAST).criteria
        assert mp["maxpro"] < min(c["maxpro"] for c in lhs)
        assert mm["maximin"] > max(c["maximin"] for c in lhs)
        assert un["wraparound_discrepancy"] < min(
            c["wraparound_discrepancy"] for c in lhs
        )

    def test_same_seed_same_design(self):
        a = sp.space_filling(10, 2, seed=5, **FAST)
        b = sp.space_filling(10, 2, seed=5, **FAST)
        np.testing.assert_array_equal(a.unit, b.unit)

    def test_numpy_fallback_agrees_in_kind(self):
        """Without the compiled kernel the search still returns a Latin
        hypercube that beats a random one."""
        from statspai.doe.spacefill import _anneal_loop, _PairCriterion, _random_lhd

        rng = np.random.default_rng(0)
        X0 = _random_lhd(10, 2, rng)
        crit = _PairCriterion("maxpro", 2, 4.0, 0.0, 10)
        with np.errstate(divide="ignore", invalid="ignore"):
            X1, f1 = _anneal_loop(X0, crit, 2000, rng)
        assert f1 < crit.total(X0)
        for col in X1.T:
            assert sorted(np.floor(col * 10).astype(int)) == list(range(10))

    def test_constraint_gives_feasible_runs(self):
        inside = lambda d: d["x1"] ** 2 + d["x2"] ** 2 <= 1  # noqa: E731
        d = sp.space_filling(15, 2, seed=1, constraint=inside)
        assert inside(d.design).all()
        assert d.n_runs == 15

    def test_constraint_that_leaves_too_little(self):
        with pytest.raises(DataInsufficient, match="satisfy the constraint"):
            sp.space_filling(20, 2, constraint=lambda d: d["x1"] < 1e-4, seed=1)

    def test_sobol_and_halton_shapes(self):
        assert sp.space_filling(9, 4, method="sobol", seed=1).design.shape == (9, 4)
        assert sp.space_filling(9, 4, method="halton", seed=1).design.shape == (9, 4)

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"n": 1, "factors": 2}, "at least two runs"),
            ({"n": 5, "factors": 2, "method": "nope"}, "method must be one of"),
            ({"n": 5, "factors": {"a": (1, 1)}}, "lower < upper"),
            ({"n": 5, "factors": 0}, "At least one factor"),
            ({"n": 5, "factors": 2, "delta": -1}, "non-negative"),
        ],
    )
    def test_bad_input(self, kwargs, match):
        with pytest.raises((MethodIncompatibility, DataInsufficient), match=match):
            sp.space_filling(**kwargs)


class TestCriteriaAndAugment:
    def test_grid_values_by_hand(self):
        grid = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
        c = sp.design_criteria(grid)
        assert c["maximin"] == pytest.approx(0.5)
        assert np.isinf(c["maxpro"])  # two runs share a level
        assert c["min_projected_distance"] == 0.0
        # farthest point of the square from the four runs: a corner
        assert c["fill_distance"] == pytest.approx(np.sqrt(2) / 4, abs=1e-12)

    def test_discrepancies_match_scipy(self):
        from scipy.stats import qmc

        X = np.random.default_rng(0).random((12, 3))
        c = sp.design_criteria(X)
        assert c["wraparound_discrepancy"] ** 2 == pytest.approx(
            qmc.discrepancy(X, method="WD"), rel=1e-12
        )
        assert c["centered_discrepancy"] ** 2 == pytest.approx(
            qmc.discrepancy(X, method="CD"), rel=1e-12
        )

    def test_bounds_rescale(self):
        X = np.random.default_rng(1).random((8, 2))
        raw = pd.DataFrame(X * [10, 4] + [5, -2], columns=["a", "b"])
        a = sp.design_criteria(X)
        b = sp.design_criteria(raw, bounds={"a": (5, 15), "b": (-2, 2)})
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-10)

    def test_off_the_cube_without_bounds(self):
        with pytest.raises(MethodIncompatibility, match="not on the unit cube"):
            sp.design_criteria(np.array([[0.0, 2.0], [1.0, 3.0]]))

    def test_augment_keeps_old_runs_and_maximin_picks_farthest(self):
        first = np.array([[0.0, 0.0], [1.0, 1.0]])
        cand = np.array([[0.5, 0.5], [1.0, 0.0], [0.9, 0.9]])
        out = sp.design_augment(first, 1, candidates=cand, criterion="maximin")
        np.testing.assert_array_equal(out.unit[:2], first)
        np.testing.assert_array_equal(out.unit[2], [1.0, 0.0])
        assert out.model_info["candidate_index"] == [1]

    def test_augment_maxpro_never_repeats_a_level(self):
        first = sp.space_filling(6, 2, seed=1, **FAST)
        out = sp.design_augment(first, 6, seed=3)
        assert np.isfinite(out.criteria["maxpro"])
        assert out.n_runs == 12

    def test_augment_errors(self):
        X = np.random.default_rng(0).random((4, 2))
        with pytest.raises(MethodIncompatibility, match="criterion"):
            sp.design_augment(X, 2, criterion="d")
        with pytest.raises(DataInsufficient, match="fewer than"):
            sp.design_augment(X, 5, candidates=np.random.default_rng(1).random((3, 2)))
        with pytest.raises(DataInsufficient, match="repeats a level"):
            sp.design_augment(X, 1, candidates=X.copy())


# ---------------------------------------------------------------------------
# Factorial designs
# ---------------------------------------------------------------------------


class TestFactorial:
    def test_full_factorials(self):
        assert sp.factorial_design(3).design.shape == (8, 3)
        d = sp.factorial_design({"price": [9, 12], "ad": ["a", "b", "c"]})
        assert d.design.shape == (6, 2)
        assert set(d.design["ad"]) == {"a", "b", "c"}
        assert sp.factorial_design(2, levels=[2, 4]).n_runs == 8

    def test_textbook_half_fraction(self):
        """2^(5-1) with E = ABCD: resolution V, one word of length five."""
        d = sp.factorial_design(5, generators=["E = ABCD"])
        assert d.resolution == 5
        assert d.defining_relation == ["ABCDE"]
        x = d.design.to_numpy()
        np.testing.assert_array_equal(x[:, 4], x[:, :4].prod(axis=1))
        # orthogonal main effects and two-factor interactions
        cols = [x[:, j] for j in range(5)] + [
            x[:, i] * x[:, j] for i, j in itertools.combinations(range(5), 2)
        ]
        M = np.column_stack(cols)
        np.testing.assert_array_equal(M.T @ M, 16 * np.eye(15))

    def test_saturated_resolution_three(self):
        """2^(7-4): every two-factor interaction aliased with a main effect."""
        d = sp.factorial_design(7, n_runs=8)
        assert d.resolution == 3
        assert d.word_length_pattern[2] == 7  # seven words of length three
        assert all(len(v) > 0 for k, v in d.aliases.items() if len(k) == 1)

    def test_resolution_four_aliases_pairs_of_interactions(self):
        d = sp.factorial_design(4, generators=["D = ABC"])
        assert d.resolution == 4
        assert d.aliases["AB"] == ["CD"]
        assert d.aliases["A"] == []

    def test_negative_generator_gives_the_other_half(self):
        plus = sp.factorial_design(3, generators=["C = AB"]).design
        minus = sp.factorial_design(3, generators=["C = -AB"]).design
        both = pd.concat([plus, minus]).drop_duplicates()
        assert both.shape[0] == 8

    def test_long_names(self):
        d = sp.factorial_design(["price", "ad", "pack"], generators=["pack = price*ad"])
        assert d.defining_relation == ["price:ad:pack"]

    def test_centre_points_replicates_and_run_order(self):
        d = sp.factorial_design(2, center_points=3, replicates=2)
        assert d.n_runs == 11
        assert (d.design.tail(3).to_numpy() == 0).all()
        r = sp.factorial_design(3, randomize=True, seed=1).design
        assert sorted(r["std_order"]) == list(range(1, 9))

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"factors": 3, "levels": 3, "n_runs": 9}, "two-level factors only"),
            ({"factors": 4, "n_runs": 6}, "power of two"),
            ({"factors": 8, "n_runs": 4}, "at most 3"),
            ({"factors": 4, "generators": ["D = ABZ"]}, "not a factor"),
            ({"factors": 4, "generators": ["D ABC"]}, "reads 'E = ABC'"),
            ({"factors": 5, "generators": ["D = ABC", "E = ABC"]}, "indistinguishable"),
            ({"factors": 12, "n_runs": 128}, "not attempted"),
            ({"factors": {"a": [1, 1]}}, "two distinct levels"),
        ],
    )
    def test_bad_input(self, kwargs, match):
        with pytest.raises(MethodIncompatibility, match=match):
            sp.factorial_design(**kwargs)

    def test_aberration_of_a_full_factorial_is_zero(self):
        a = sp.design_aberration(sp.factorial_design(3, levels=3).design)
        assert np.allclose([a["A1"], a["A2"], a["A3"]], 0, atol=1e-10)
        assert np.isnan(a["resolution"])

    def test_aberration_sees_imbalance(self):
        d = pd.DataFrame({"a": [0, 0, 0, 1], "b": [0, 1, 0, 1]})
        a = sp.design_aberration(d)
        assert a["A1"] > 0 and a["resolution"] == 1

    def test_aberration_errors(self):
        with pytest.raises(MethodIncompatibility, match="single level"):
            sp.design_aberration(pd.DataFrame({"a": [1, 1], "b": [0, 1]}))
        with pytest.raises(MethodIncompatibility, match="max_order"):
            sp.design_aberration(sp.factorial_design(2).design, max_order=5)


class TestFactorialEffects:
    @staticmethod
    def _data(seed=0, reps=1, sd=0.0):
        d = sp.factorial_design(4, replicates=reps).design
        rng = np.random.default_rng(seed)
        d["y"] = (
            20
            + 4 * d["A"]
            - 3 * d["C"]
            + 2 * d["A"] * d["C"]
            + sd * rng.standard_normal(len(d))
        )
        return d

    def test_noise_free_effects_are_exact(self):
        fit = sp.factorial_effects(self._data(), "y")
        eff = fit.effects["effect"]
        assert eff["A"] == pytest.approx(8.0)
        assert eff["C"] == pytest.approx(-6.0)
        assert eff["AC"] == pytest.approx(4.0)
        assert np.allclose(eff.drop(["A", "C", "AC"]), 0, atol=1e-12)
        assert fit.intercept == pytest.approx(20.0)

    def test_active_effects_are_found_in_noise(self):
        fit = sp.factorial_effects(self._data(sd=0.5), "y")
        assert set(fit.active) >= {"A", "C", "AC"}
        assert fit.df_resid == 0
        assert "se" not in fit.effects.columns

    def test_replicates_give_ordinary_inference(self):
        d = self._data(seed=3, reps=2, sd=1.0)
        fit = sp.factorial_effects(d, "y")
        ols = sp.regress("y ~ A*B*C*D", d)
        assert fit.df_resid == 16
        assert fit.effects.loc["A", "se"] == pytest.approx(
            2 * ols.std_errors["A"], rel=1e-10
        )
        assert fit.effects.loc["A", "t"] == pytest.approx(ols.tvalues["A"], rel=1e-10)

    def test_lenth_size_under_the_null(self):
        """With no real effect the individual margin flags about 5% of
        effects and the simultaneous margin about 5% of experiments. 400
        experiments of 15 effects: binomial sd 0.3% and 1.1%."""
        from statspai.doe.effects import _lenth_null, lenth_pse

        null_all, null_max = _lenth_null(15, 20000, 0)
        me, sme = np.quantile(null_all, 0.95), np.quantile(null_max, 0.95)
        rng = np.random.default_rng(42)
        ind, fam = [], []
        for _ in range(400):
            e = rng.standard_normal(15)
            t = np.abs(e) / lenth_pse(e)
            ind.append(np.mean(t > me))
            fam.append(np.any(t > sme))
        assert abs(np.mean(ind) - 0.05) < 0.012
        assert abs(np.mean(fam) - 0.05) < 0.035

    def test_any_two_level_coding(self):
        d = self._data(sd=0.3)
        relab = d.assign(A=d["A"].map({-1: "low", 1: "zhigh"}), C=d["C"] * 7 + 3)
        a = sp.factorial_effects(d, "y").effects["effect"]
        b = sp.factorial_effects(relab, "y").effects["effect"]
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=1e-10)

    def test_plot(self):
        import matplotlib

        matplotlib.use("Agg")
        ax = sp.factorial_effects(self._data(sd=0.5), "y").plot()
        assert ax.get_xlabel() == "absolute effect"

    def test_errors(self):
        d = self._data()
        with pytest.raises(MethodIncompatibility, match="exactly two"):
            sp.factorial_effects(d.assign(A=[0, 1, 2, 3] * 4), "y", factors=["A", "B"])
        with pytest.raises(Exception, match="not in the data"):
            sp.factorial_effects(d, "nope")
        with pytest.raises(MethodIncompatibility, match="reference"):
            sp.factorial_effects(d, "y", reference="z")
        with pytest.raises(MethodIncompatibility, match="order"):
            sp.factorial_effects(d, "y", order=9)


# ---------------------------------------------------------------------------
# Mixture designs
# ---------------------------------------------------------------------------


class TestMixture:
    @pytest.mark.parametrize("q,m", [(3, 2), (3, 3), (4, 2), (5, 3)])
    def test_lattice_count_and_sums(self, q, m):
        from math import comb

        d = sp.mixture_design(q, degree=m)
        assert d.n_runs == comb(q + m - 1, m)
        np.testing.assert_allclose(d.design.sum(axis=1), 1.0)
        assert np.allclose((d.design.to_numpy() * m) % 1, 0)

    def test_centroid(self):
        d = sp.mixture_design(4, kind="simplex_centroid")
        assert d.n_runs == 15
        assert (d.design.round(10) == 0.25).all(axis=1).sum() == 1

    def test_lower_bounds_and_total(self):
        d = sp.mixture_design(3, degree=2, lower=[10, 20, 30], total=100)
        np.testing.assert_allclose(d.design.sum(axis=1), 100.0)
        assert (d.design.to_numpy() >= np.array([10, 20, 30]) - 1e-9).all()

    def test_space_filling_under_a_constraint(self):
        ok = lambda f: (f["x1"] + f["x2"] < 0.7) & (
            f["x1"] + f["x2"] > 0.3
        )  # noqa: E731
        d = sp.mixture_design(3, kind="space_filling", n=9, constraint=ok, seed=1)
        np.testing.assert_allclose(d.design.sum(axis=1), 1.0)
        assert ok(d.design).all()
        assert (d.design.to_numpy() >= 0).all()

    def test_errors(self):
        with pytest.raises(MethodIncompatibility, match="two distinct"):
            sp.mixture_design(1)
        with pytest.raises(MethodIncompatibility, match="n >= 2"):
            sp.mixture_design(3, kind="space_filling")
        with pytest.raises(MethodIncompatibility, match="kind must be"):
            sp.mixture_design(3, kind="x")
        with pytest.raises(MethodIncompatibility, match="lower"):
            sp.mixture_design(3, lower=[0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# Optimal designs: closed forms
# ---------------------------------------------------------------------------


class TestOptimalDesign:
    @pytest.mark.parametrize("degree", [1, 2, 3, 4])
    def test_polynomial_regression_closed_form(self, degree):
        """D-optimal design for a polynomial of degree d on [-1, 1]: equal
        weight on the d + 1 roots of (1 - x^2) P_d'(x), with P_d the
        Legendre polynomial (bib key ``fedorov1972theory``)."""
        formula = " + ".join(f"I(x**{j})" for j in range(1, degree + 1))
        d = sp.doe_optimal(formula, {"x": (-1, 1)})
        inner = legendre.Legendre.basis(degree).deriv().roots()
        truth = np.sort(np.r_[-1.0, inner, 1.0])
        # the polish moves the points with a finite-difference gradient
        np.testing.assert_allclose(d.design["x"], truth, atol=2e-5)
        np.testing.assert_allclose(d.design["weight"], 1 / (degree + 1), atol=2e-4)
        # a lower bound, first-order sensitive to the 1e-5 left by the polish
        assert d.efficiency > 0.9995

    def test_quadratic_a_optimal_closed_form(self):
        """A-optimal for a quadratic on [-1, 1]: weights 1/4, 1/2, 1/4."""
        d = sp.doe_optimal("x + I(x**2)", {"x": (-1, 1)}, criterion="A")
        np.testing.assert_allclose(d.design["x"], [-1, 0, 1], atol=1e-5)
        np.testing.assert_allclose(d.design["weight"], [0.25, 0.5, 0.25], atol=1e-4)

    def test_exponential_decay_locally_optimal(self):
        """a exp(-b t): observe at 0 and at 1 / b. The determinant of the
        two-point information matrix is proportional to
        ``(t2 - t1)^2 exp(-2 b (t1 + t2))``, maximised at t1 = 0, t2 = 1 / b."""
        d = sp.doe_optimal(
            "{a} * exp(-{b} * t)", {"t": (0, 10)}, params={"a": 1, "b": 0.5}
        )
        np.testing.assert_allclose(d.design["t"], [0.0, 2.0], atol=1e-4)
        np.testing.assert_allclose(d.design["weight"], 0.5, atol=1e-4)

    def test_two_parameter_logistic(self):
        """Logit with two parameters: equal weight where the linear
        predictor is +-1.5434 (the root of e^u = (u + 1) / (u - 1))."""
        from scipy.optimize import brentq

        u = brentq(lambda v: np.exp(v) - (v + 1) / (v - 1), 1.1, 3)
        d = sp.doe_optimal(
            "{b0} + {b1} * x",
            {"x": (-10, 10)},
            params={"b0": 1, "b1": 2},
            family="binomial",
        )
        np.testing.assert_allclose(
            d.design["x"], [(-u - 1) / 2, (u - 1) / 2], atol=1e-3
        )
        np.testing.assert_allclose(d.design["weight"], 0.5, atol=1e-3)

    def test_exact_design_of_a_two_by_two_model(self):
        d = sp.doe_optimal("a * b", {"a": (0, 1), "b": (0, 1)}, n=12, seed=1)
        assert d.design["n"].tolist() == [3, 3, 3, 3]
        assert d.efficiency == pytest.approx(1.0, abs=1e-6)
        assert d.to_frame().shape == (12, 2)

    def test_prior_spreads_the_support(self):
        local = sp.doe_optimal(
            "{a} * exp(-{b} * t)", {"t": (0, 10)}, params={"a": 1, "b": 0.5}
        )
        bayes = sp.doe_optimal(
            "{a} * exp(-{b} * t)",
            {"t": (0, 10)},
            params={"a": 1},
            prior={"b": (0.1, 2.0)},
            seed=1,
        )
        assert bayes.design.shape[0] > local.design.shape[0]
        assert bayes.efficiency > 0.995

    def test_first_order_model_in_four_factors(self):
        """Any design with orthogonal columns at the corners is D-optimal:
        the information matrix per run is the identity, criterion 1. The
        support (16 corners) is too large to move, so only the weights are
        refined; the bound must still certify the design."""
        d = sp.doe_optimal(
            "x1 + x2 + x3 + x4", {f"x{i}": (-1, 1) for i in range(1, 5)}, grid=3
        )
        assert d.criterion_value == pytest.approx(1.0, abs=1e-3)
        assert d.efficiency > 0.999
        assert set(np.abs(d.design[["x1", "x2", "x3", "x4"]].to_numpy()).ravel()) == {
            1.0
        }

    def test_grid_too_large(self):
        with pytest.raises(MethodIncompatibility, match="candidates"):
            sp.doe_optimal(
                "x1 + x2 + x3 + x4 + x5 + x6",
                {f"x{i}": (-1, 1) for i in range(1, 7)},
                grid=11,
            )

    def test_qualitative_candidates(self):
        cand = pd.DataFrame(
            list(itertools.product(["a", "b", "c"], [0.0, 0.5, 1.0])),
            columns=["g", "x"],
        )
        d = sp.doe_optimal("C(g) + x", candidates=cand, n=6, seed=1)
        assert d.n_runs == 6
        assert set(d.to_frame()["x"]) == {0.0, 1.0}

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"model": "x", "factors": {"x": (0, 1)}, "criterion": "G"}, "criterion"),
            ({"model": "x"}, "either factors= or candidates="),
            ({"model": "{a}*x", "factors": {"x": (0, 1)}}, "No value for a"),
            ({"model": "x", "factors": {"x": (0, 1)}, "params": {"a": 1}}, "braces"),
            ({"model": "x", "factors": {"x": (0, 1)}, "family": "binomial"}, "braces"),
            (
                {"model": "x + I(x**2)", "factors": {"x": (0, 1)}, "n": 2},
                "at least 3",
            ),
            (
                {
                    "model": "{a}*x",
                    "factors": {"x": (0, 1)},
                    "params": {"a": 1, "z": 2},
                },
                "Not parameters",
            ),
            (
                {
                    "model": "x + x2",
                    "candidates": pd.DataFrame({"x": [0, 1], "x2": [0, 1]}),
                },
                "not linearly independent",
            ),
        ],
    )
    def test_bad_input(self, kwargs, match):
        with pytest.raises((MethodIncompatibility, DataInsufficient), match=match):
            sp.doe_optimal(**kwargs)


# ---------------------------------------------------------------------------
# Sensitivity analysis: functions with known indices
# ---------------------------------------------------------------------------


class TestSobol:
    def test_additive_function(self):
        """y = x1 + 2 x2 with uniform inputs: shares 1/5 and 4/5, x3 inert."""
        res = sp.sobol_indices(
            lambda X: X[:, 0] + 2 * X[:, 1],
            3,
            n=4096,
            pass_as="array",
            seed=0,
            n_boot=0,
        )
        np.testing.assert_allclose(res.indices["first"], [0.2, 0.8, 0.0], atol=0.01)
        np.testing.assert_allclose(res.indices["total"], [0.2, 0.8, 0.0], atol=0.01)
        assert res.variance == pytest.approx(5 / 12, rel=0.02)

    @pytest.mark.parametrize("estimator", ["jansen", "saltelli"])
    def test_ishigami(self, estimator):
        """Ishigami function, a = 7, b = 0.1, inputs uniform on (-pi, pi)."""
        a, b = 7.0, 0.1
        f = lambda d: (  # noqa: E731
            np.sin(d["x1"])
            + a * np.sin(d["x2"]) ** 2
            + b * d["x3"] ** 4 * np.sin(d["x1"])
        )
        V = a**2 / 8 + b * np.pi**4 / 5 + b**2 * np.pi**8 / 18 + 0.5
        v1 = 0.5 * (1 + b * np.pi**4 / 5) ** 2
        v2 = a**2 / 8
        v13 = b**2 * np.pi**8 * (1 / 18 - 1 / 50)
        first = np.array([v1, v2, 0.0]) / V
        total = np.array([v1 + v13, v2, v13]) / V
        res = sp.sobol_indices(
            f,
            {k: (-np.pi, np.pi) for k in ("x1", "x2", "x3")},
            n=8192,
            estimator=estimator,
            seed=1,
            n_boot=0,
        )
        np.testing.assert_allclose(res.indices["first"], first, atol=0.02)
        np.testing.assert_allclose(res.indices["total"], total, atol=0.02)

    def test_non_uniform_inputs(self):
        """y = x1 + x2 with sd 1 and 3: shares 1/10 and 9/10."""
        res = sp.sobol_indices(
            lambda d: d["a"] + d["b"],
            {"a": stats.norm(0, 1), "b": stats.norm(5, 3)},
            n=4096,
            seed=2,
            n_boot=0,
        )
        np.testing.assert_allclose(res.indices["first"], [0.1, 0.9], atol=0.02)

    def test_intervals_cover_with_random_sampling(self):
        f = lambda X: X[:, 0] + 2 * X[:, 1] + X[:, 0] * X[:, 2]  # noqa: E731
        res = sp.sobol_indices(
            f, 3, n=2000, sampling="random", pass_as="array", seed=3, n_boot=200
        )
        tab = res.indices
        assert (tab["total_lower"] <= tab["total"]).all()
        assert (tab["total"] <= tab["total_upper"]).all()
        assert res.n_evaluations == 2000 * 5

    def test_errors(self):
        with pytest.raises(MethodIncompatibility, match="does not vary"):
            sp.sobol_indices(lambda X: np.ones(len(X)), 2, n=64, pass_as="array")
        with pytest.raises(MethodIncompatibility, match="one number per run"):
            sp.sobol_indices(lambda X: X, 2, n=64, pass_as="array")
        with pytest.raises(MethodIncompatibility, match="missing or infinite"):
            sp.sobol_indices(lambda X: np.log(X[:, 0] - 0.5), 2, n=64, pass_as="array")
        with pytest.raises(MethodIncompatibility, match="both A and B"):
            sp.sobol_indices(lambda X: X[:, 0], 2, A=np.zeros((4, 2)), pass_as="array")
        with pytest.raises(MethodIncompatibility, match="estimator"):
            sp.sobol_indices(lambda X: X[:, 0], 2, estimator="x")


class TestMorris:
    def test_linear_function_has_constant_effects(self):
        res = sp.morris_screening(
            lambda X: 3 * X[:, 0] - 2 * X[:, 1] + 0 * X[:, 2],
            3,
            r=6,
            pass_as="array",
            seed=0,
        )
        np.testing.assert_allclose(res.effects["mu"], [3, -2, 0], atol=1e-10)
        np.testing.assert_allclose(res.effects["mu_star"], [3, 2, 0], atol=1e-10)
        np.testing.assert_allclose(res.effects["sigma"], 0, atol=1e-10)
        assert res.n_evaluations == 6 * 4

    def test_effects_are_per_full_range(self):
        res = sp.morris_screening(
            lambda d: 5 * d["a"], {"a": (0, 10), "b": (0, 1)}, r=4, seed=1
        )
        assert res.effects.loc["a", "mu"] == pytest.approx(50.0)

    def test_interaction_shows_in_sigma(self):
        res = sp.morris_screening(
            lambda X: X[:, 0] * X[:, 1] + X[:, 2], 3, r=30, pass_as="array", seed=2
        )
        assert res.effects["sigma"].iloc[0] > 0.1
        assert res.effects["sigma"].iloc[2] == pytest.approx(0, abs=1e-10)

    def test_trajectories_stay_in_the_cube_on_the_grid(self):
        res = sp.morris_screening(
            lambda X: X.sum(axis=1), 5, r=20, levels=6, pass_as="array", seed=3
        )
        U = res.design.to_numpy()
        assert U.min() >= 0 and U.max() <= 1
        assert np.allclose((U * 5) % 1, 0, atol=1e-9) or np.allclose(
            np.round(U * 5), U * 5, atol=1e-9
        )

    def test_errors(self):
        with pytest.raises(MethodIncompatibility, match="one-factor-at-a-time"):
            sp.morris_screening(None, 2, design=np.random.rand(6, 2), y=np.zeros(6))
        with pytest.raises(MethodIncompatibility, match="distributions"):
            sp.morris_screening(lambda d: d["a"], {"a": stats.norm()})
        with pytest.raises(MethodIncompatibility, match="Give func"):
            sp.morris_screening(None, 2)


# ---------------------------------------------------------------------------
# Representative points
# ---------------------------------------------------------------------------


class TestSupportPoints:
    def test_one_dimensional_uniform(self):
        """Support points of U(0, 1) are the midpoints (2i - 1) / (2n)."""
        y = (np.arange(20000) + 0.5) / 20000
        res = sp.support_points(pd.DataFrame({"x": y}), 5, seed=0, tol=1e-7)
        np.testing.assert_allclose(
            np.sort(res.points["x"]), (2 * np.arange(1, 6) - 1) / 10, atol=2e-3
        )

    def test_integration_beats_random_subsamples(self):
        rng = np.random.default_rng(0)
        z = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=4000)
        df = pd.DataFrame(z, columns=["a", "b"])
        g = lambda d: np.exp(0.5 * d["a"] - 0.3 * d["b"])  # noqa: E731
        truth = g(df).mean()
        res = sp.support_points(df, 30, seed=1)
        err_sp = abs(g(res.points).mean() - truth)
        err_mc = np.mean(
            [abs(g(df.sample(30, random_state=s)).mean() - truth) for s in range(50)]
        )
        assert err_sp < 0.25 * err_mc
        assert res.energy < res.energy_random

    def test_subsample_returns_rows(self):
        df = pd.DataFrame(
            np.random.default_rng(2).normal(size=(500, 3)), columns=list("abc")
        )
        res = sp.support_points(df, 20, subsample=True, seed=1)
        assert len(set(res.index)) == 20
        np.testing.assert_array_equal(
            res.points.to_numpy(), df.iloc[res.index].to_numpy()
        )

    def test_distributions(self):
        res = sp.support_points(
            {"r": stats.norm(0.1, 0.016), "k": stats.uniform(990, 120)}, 25, seed=1
        )
        assert res.points["k"].between(990, 1110).all()
        assert res.points["r"].mean() == pytest.approx(0.1, abs=0.002)

    def test_weights_shift_the_points(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"x": rng.uniform(0, 1, 3000)})
        w = df["x"].to_numpy() ** 3
        res = sp.support_points(df, 10, weights=w, seed=1)
        assert res.points["x"].mean() > 0.7  # mean of Beta(4, 1) is 0.8

    def test_errors(self):
        df = pd.DataFrame({"x": np.arange(10.0), "g": list("ab") * 5})
        with pytest.raises(MethodIncompatibility, match="Non-numeric"):
            sp.support_points(df, 2)
        with pytest.raises(DataInsufficient, match="too few"):
            sp.support_points(df[["x"]], 8)
        with pytest.raises(MethodIncompatibility, match="Constant"):
            sp.support_points(pd.DataFrame({"x": np.ones(50)}), 5)
        with pytest.raises(MethodIncompatibility, match="frozen"):
            sp.support_points({"x": 3}, 5)


class TestSplitData:
    @staticmethod
    def _df(n=300, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        return pd.DataFrame(
            {"x": x, "y": x**2 + rng.normal(size=n), "g": rng.choice(list("abc"), n)}
        )

    @pytest.mark.parametrize("method", ["support", "twinning"])
    def test_partition(self, method):
        df = self._df()
        split = sp.split_data(df, 0.2, method=method, seed=1)
        train, test = split
        assert len(test) == 60 and len(train) == 240
        assert set(train.index).isdisjoint(test.index)
        assert set(train.index) | set(test.index) == set(df.index)
        assert split.energy < split.energy_random

    def test_twinning_is_deterministic_given_start(self):
        df = self._df()
        a = sp.split_data(df, 0.25, method="twinning", start=7)
        b = sp.split_data(df, 0.25, method="twinning", start=7)
        np.testing.assert_array_equal(a.test_index, b.test_index)
        assert 7 in a.test_index

    def test_test_error_is_less_variable_than_with_random_splits(self):
        """The point of the method: the test error of a fixed model varies
        less across splits. A cubic fit to a noisy sine, 40 splits each."""
        rng = np.random.default_rng(5)
        x = rng.uniform(-2, 2, 400)
        df = pd.DataFrame({"x": x, "y": np.sin(2 * x) + 0.3 * rng.normal(size=400)})

        def test_rmse(train, test):
            c = np.polyfit(train["x"], train["y"], 3)
            return float(np.sqrt(np.mean((np.polyval(c, test["x"]) - test["y"]) ** 2)))

        opt = [
            test_rmse(*sp.split_data(df, 0.2, method="support", seed=s))
            for s in range(12)
        ]
        rnd = []
        for s in range(40):
            te = df.sample(frac=0.2, random_state=s)
            rnd.append(test_rmse(df.drop(te.index), te))
        assert np.std(opt) < 0.6 * np.std(rnd)

    def test_auto_picks_twinning_for_large_data(self):
        df = self._df(n=2500)[["x", "y"]]
        assert sp.split_data(df, 0.2, seed=1).model_info["method"] == "twinning"
        assert "SPlit" in sp.split_data(df.head(500), 0.2, seed=1).model_info["method"]

    def test_errors(self):
        df = self._df()
        with pytest.raises(MethodIncompatibility, match="whole number"):
            sp.split_data(df, 0.3, method="twinning")
        with pytest.raises(MethodIncompatibility, match="test_size"):
            sp.split_data(df, 1.2)
        with pytest.raises(MethodIncompatibility, match="missing"):
            sp.split_data(df.assign(x=np.nan), 0.2)
        with pytest.raises(MethodIncompatibility, match="start applies"):
            sp.split_data(df, 0.2, method="support", start=3)


# ---------------------------------------------------------------------------
# Gaussian process additions and sequential design
# ---------------------------------------------------------------------------


class TestGaussianProcess:
    @staticmethod
    def _df():
        x = (np.arange(1, 9) - 0.5) / 8
        f = np.sin(10 * np.pi * x) / (1 + 64 * (x - 0.25) ** 2) + x**2
        return pd.DataFrame({"x": x, "y": f})

    def test_interpolation_passes_through_the_data(self):
        df = self._df()
        fit = sp.gp_regress("y ~ x", df, interpolate=True, seed=0)
        pred = fit.predict(df)
        np.testing.assert_allclose(pred["mean"], df["y"], atol=1e-5)
        assert pred["sd"].max() < 1e-3
        assert fit.params["noise_var"] < 1e-6

    def test_profile_likelihood_optimum(self):
        """At the ML optimum the process variance is r' R^-1 r / n, at the
        REML optimum r' R^-1 r / (n - 1) (the two estimating equations)."""
        df = self._df()
        n = len(df)
        for lik, div in (("ml", n), ("reml", n - 1)):
            fit = sp.gp_regress("y ~ x", df, interpolate=True, likelihood=lik, seed=0)
            ls, sf, mu = fit.params[["length_scale[x]", "signal_var", "mean"]]
            d = df["x"].to_numpy()[:, None] - df["x"].to_numpy()[None, :]
            R = np.exp(-0.5 * (d / ls) ** 2) + (fit.params["noise_var"] / sf) * np.eye(
                n
            )
            r = df["y"].to_numpy() - mu
            assert sf == pytest.approx(r @ np.linalg.solve(R, r) / div, rel=2e-4)

    def test_expected_improvement_formula(self):
        df = self._df()
        fit = sp.gp_regress("y ~ x", df, interpolate=True, seed=0)
        new = pd.DataFrame({"x": np.linspace(0, 1, 41)})
        pred = fit.predict(new)
        u = (df["y"].min() - pred["mean"]) / pred["sd"].clip(lower=1e-12)
        expect = pred["sd"].clip(lower=1e-12) * (
            u * stats.norm.cdf(u) + stats.norm.pdf(u)
        )
        np.testing.assert_allclose(fit.expected_improvement(new), expect, atol=1e-12)
        up = fit.expected_improvement(new, minimize=False)
        assert (up >= 0).all() and up.idxmax() != fit.expected_improvement(new).idxmax()

    def test_short_length_scale_is_reported(self):
        x = np.repeat(np.arange(6.0), 2)
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": x, "y": rng.normal(size=12)})
        with pytest.warns(Warning, match="says nothing between the data points"):
            sp.gp_regress("y ~ x", df, length_scale=0.01, optimize_hyper=False)

    def test_errors(self):
        df = self._df()
        with pytest.raises(MethodIncompatibility, match="likelihood"):
            sp.gp_regress("y ~ x", df, likelihood="x")
        with pytest.raises(MethodIncompatibility, match="do not pass noise_var"):
            sp.gp_regress("y ~ x", df, interpolate=True, noise_var=0.1)


class TestSequentialDesign:
    def test_minimises_a_multimodal_function(self):
        """True minimum -0.60348 at x = 0.1578 (grid search)."""
        f = lambda d: (  # noqa: E731
            np.sin(10 * np.pi * d["x"]) / (1 + 64 * (d["x"] - 0.25) ** 2) + d["x"] ** 2
        )
        start = pd.DataFrame({"x": (np.arange(1, 9) - 0.5) / 8})
        res = sp.sequential_design(
            f, {"x": (0, 1)}, n_new=8, design=start, y=f(start), seed=1
        )
        assert res.best["y"] < -0.60
        assert abs(res.best["x"] - 0.1578) < 0.01
        assert res.design.shape[0] == 16
        assert (res.design["stage"] == "sequential").sum() == 8

    def test_maximise_and_predict(self):
        f = lambda d: -((d["a"] - 2) ** 2) - (d["b"] + 1) ** 2  # noqa: E731
        res = sp.sequential_design(
            f, {"a": (0, 4), "b": (-3, 1)}, n_new=10, goal="maximize", seed=2
        )
        assert res.best["y"] > -0.05
        pred = res.predict(pd.DataFrame({"a": [2.0], "b": [-1.0]}))
        assert pred["mean"].iloc[0] == pytest.approx(0.0, abs=0.05)

    def test_emulation_reduces_uncertainty(self):
        f = lambda d: np.sin(6 * d["x"])  # noqa: E731
        res = sp.sequential_design(
            f, {"x": (0, 1)}, n_new=6, n_init=5, goal="emulate", seed=3
        )
        assert res.best == {}
        assert res.trace["criterion"].iloc[-1] < res.trace["criterion"].iloc[0]

    def test_errors(self):
        f = lambda d: d["x"]  # noqa: E731
        with pytest.raises(MethodIncompatibility, match="goal"):
            sp.sequential_design(f, {"x": (0, 1)}, goal="x")
        with pytest.raises(MethodIncompatibility, match="dict"):
            sp.sequential_design(f, 2)
        with pytest.raises(MethodIncompatibility, match="same value"):
            sp.sequential_design(
                lambda d: np.zeros(len(d)), {"x": (0, 1)}, n_new=2, seed=1
            )


# ---------------------------------------------------------------------------
# Inputs an independent review found mishandled (2026-10-07)
# ---------------------------------------------------------------------------


class TestReviewFindings:
    def test_negative_generator_carries_its_sign(self):
        """D = -AB: on the fraction ABD is -1, so A is aliased with -BD and
        the main-effect column of A estimates A - BD + CE."""
        d = sp.factorial_design(5, generators=["D = -AB", "E = AC"])
        x = d.design
        assert (x["A"] * x["B"] * x["D"] == -1).all()
        assert (x["A"] * x["C"] * x["E"] == 1).all()
        assert d.defining_relation == ["-ABD", "ACE", "-BCDE"]
        assert d.aliases["A"] == ["-BD", "CE"]
        assert d.aliases["D"] == ["-AB"]
        # every listed alias is the stated multiple of the column
        cols = {nm: x[nm].to_numpy() for nm in x.columns}

        def column(word):
            return np.prod([cols[c] for c in word.lstrip("-")], axis=0)

        for effect, mates in d.aliases.items():
            for m in mates:
                sign = -1 if m.startswith("-") else 1
                np.testing.assert_array_equal(column(effect), sign * column(m))

    def test_generators_with_spaces_in_names(self):
        d = sp.factorial_design(
            ["x 1", "x 2", "x 3", "x 4"], generators=["x 4 = x 1*x 2*x 3"]
        )
        assert d.resolution == 4 and d.n_runs == 8

    def test_empty_generator_list_is_a_full_factorial(self):
        assert sp.factorial_design(3, generators=[]).n_runs == 8

    def test_effects_with_integer_column_names_and_seed_none(self):
        d = sp.factorial_design(3).design
        d.columns = [0, 1, 2]
        d["y"] = 2.0 * d[0] - d[2]
        fit = sp.factorial_effects(d, "y", seed=None)
        assert fit.effects.loc["0", "effect"] == pytest.approx(4.0)

    def test_effects_refuse_centre_points_with_advice(self):
        d = sp.factorial_design(2, center_points=3).design
        d["y"] = np.arange(len(d), dtype=float)
        with pytest.raises(MethodIncompatibility, match="centre points"):
            sp.factorial_effects(d, "y")

    @pytest.mark.parametrize("p", [8, 9])
    def test_first_order_design_with_many_factors(self, p):
        """2^p corners with weight 2^-p each: below any fixed weight
        threshold. Crashed before the thresholds were made relative."""
        names = [f"x{i}" for i in range(p)]
        d = sp.doe_optimal(" + ".join(names), {nm: (-1, 1) for nm in names})
        assert d.criterion_value == pytest.approx(1.0, abs=1e-6)
        assert d.efficiency > 0.999

    def test_i_criterion_averages_over_the_box(self):
        """First-order model in five factors on [-1, 1]^5: with the corner
        design the prediction variance at x is 1 + |x|^2, whose mean over
        the cube is 1 + 5 / 3."""
        names = [f"x{i}" for i in range(5)]
        d = sp.doe_optimal(
            " + ".join(names), {nm: (-1, 1) for nm in names}, criterion="I", seed=1
        )
        assert d.criterion_value == pytest.approx(1 + 5 / 3, rel=1e-4)

    def test_i_optimal_quadratic_does_not_depend_on_the_grid(self):
        """I-optimal for a quadratic on [-1, 1]: weights 1/4, 1/2, 1/4."""
        d = sp.doe_optimal(
            "x + I(x**2)", {"x": (-1, 1)}, criterion="I", grid=21, seed=1
        )
        np.testing.assert_allclose(d.design["x"], [-1, 0, 1], atol=1e-4)
        np.testing.assert_allclose(d.design["weight"], [0.25, 0.5, 0.25], atol=2e-3)

    def test_i_criterion_of_a_poisson_model_is_the_linear_predictor_variance(self):
        d = sp.doe_optimal(
            "{b0} + {b1} * x",
            {"x": (0, 10)},
            params={"b0": 1, "b1": -1},
            family="poisson",
            criterion="I",
            seed=1,
        )
        x, w = d.design["x"].to_numpy(), d.design["weight"].to_numpy()
        X = np.column_stack([np.ones_like(x), x])
        M = (X * (w * np.exp(1 - x))[:, None]).T @ X
        g = np.linspace(0, 10, 20001)
        F = np.column_stack([np.ones_like(g), g])
        by_hand = np.mean(np.einsum("ij,jk,ik->i", F, np.linalg.inv(M), F))
        assert d.criterion_value == pytest.approx(by_hand, rel=1e-3)

    def test_nonlinear_model_rejects_names_the_expression_cannot_read(self):
        """``_n`` is the row number of the expression language and a hyphen
        is a minus sign; both used to give a silent wrong design."""
        for name in ("_n", "price-usd"):
            with pytest.raises(MethodIncompatibility, match="plain identifiers"):
                sp.doe_optimal("{a} + {b}", {name: (0, 1)}, params={"a": 1, "b": 1})

    def test_too_many_factors_for_a_box(self):
        names = [f"x{i}" for i in range(11)]
        with pytest.raises(MethodIncompatibility, match="candidates="):
            sp.doe_optimal(" + ".join(names), {nm: (-1, 1) for nm in names})

    def test_sequential_design_reserved_factor_names(self):
        with pytest.raises(MethodIncompatibility, match="cannot be called 'y'"):
            sp.sequential_design(lambda d: d["y"], {"y": (0, 1)}, n_new=1)

    def test_support_points_with_few_positive_weights(self):
        df = pd.DataFrame({"x": np.arange(100.0)})
        w = np.r_[np.ones(5), np.zeros(95)]
        with pytest.raises(DataInsufficient, match="positive weight"):
            sp.support_points(df, 10, weights=w)

    def test_energy_distance_matches_columns_by_name(self):
        rng = np.random.default_rng(0)
        sample = pd.DataFrame(
            {"a": rng.uniform(size=500), "b": rng.uniform(size=500) ** 3}
        )
        d = sp.space_filling(8, ["a", "b"], method="lhs", seed=1)
        one = sp.design_criteria(d, target=sample)["energy_distance"]
        two = sp.design_criteria(d, target=sample[["b", "a"]])["energy_distance"]
        assert one == pytest.approx(two, rel=1e-12)

    def test_morris_needs_two_trajectories(self):
        X = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]])
        with pytest.raises(DataInsufficient, match="two trajectories"):
            sp.morris_screening(None, 2, design=X, y=[0.0, 1.0, 2.0])

    def test_morris_default_jump_visits_levels_equally_when_even(self):
        res = sp.morris_screening(
            lambda X: X.sum(axis=1), 3, r=600, levels=4, pass_as="array", seed=0
        )
        counts = np.unique(np.round(res.design.to_numpy(), 6), return_counts=True)[1]
        assert counts.max() / counts.min() < 1.15


# ---------------------------------------------------------------------------
# Registry and result protocol
# ---------------------------------------------------------------------------

PUBLIC = [
    "space_filling",
    "design_augment",
    "design_criteria",
    "factorial_design",
    "design_aberration",
    "factorial_effects",
    "mixture_design",
    "doe_optimal",
    "sobol_indices",
    "morris_screening",
    "support_points",
    "split_data",
    "sequential_design",
]


@pytest.mark.parametrize("name", PUBLIC)
def test_registered_with_every_parameter(name):
    import inspect

    assert name in sp.list_functions()
    schema = sp.function_schema(name)["parameters"]["properties"]
    assert set(inspect.signature(getattr(sp, name)).parameters) == set(schema)


def test_results_serialise():
    import json

    objs = [
        sp.space_filling(6, 2, method="lhs", seed=1),
        sp.factorial_design(4, n_runs=8),
        sp.factorial_effects(TestFactorialEffects._data(sd=0.2), "y"),
        sp.doe_optimal("x", {"x": (0, 1)}),
        sp.morris_screening(lambda X: X[:, 0], 2, r=3, pass_as="array", seed=1),
    ]
    for obj in objs:
        json.dumps(obj.to_dict())
        assert isinstance(obj.summary(), str)
