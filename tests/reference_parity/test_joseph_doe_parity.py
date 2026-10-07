"""``statspai.doe`` against the R packages used in Joseph (2025),
*Experimental Design for Data Science and Engineering*.

The reference numbers are in ``_fixtures/joseph_doe_R.json``, written by
``_fixtures/_generate_joseph_doe.R`` from public test functions and data
simulated there; the same data are read back from the JSON.

Three kinds of comparison, kept apart:

* **Deterministic quantities** (criteria of a given design, greedy
  augmentation from given candidates, word length patterns, twinning from a
  given start, Sobol' and Morris statistics from given runs, kriging
  predictions at given hyperparameters). Tolerances are rounding error.
* **Optimisers of the same objective** (exact optimal designs, kriging
  likelihood). The criterion value is compared; where both sides reach the
  optimum it agrees to the optimiser's tolerance.
* **Stochastic searches** (space-filling designs, support points, SPlit,
  Lenth's simulated margins). Only a screen: our criterion must be on par
  with the reference. These are not parity claims.

All reference packages are GPL or LGPL. Nothing was translated; the
implementations follow the papers and were compared with R as a black box.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.doe.support import _nearest_distinct, helmert_frame, twin_indices

REF = json.loads(
    (Path(__file__).parent / "_fixtures" / "joseph_doe_R.json").read_text(
        encoding="utf-8"
    )
)

LOWER = np.array([0.05, 100, 63070, 990, 63.1, 700, 1120, 9855])
UPPER = np.array([0.15, 50000, 115600, 1110, 116, 820, 1680, 12045])


def borehole(U: np.ndarray) -> np.ndarray:
    x = LOWER + np.asarray(U) * (UPPER - LOWER)
    lg = np.log(x[:, 1] / x[:, 0])
    return (
        2
        * np.pi
        * x[:, 2]
        * (x[:, 3] - x[:, 5])
        / (
            lg
            * (
                1
                + 2 * x[:, 6] * x[:, 2] / (lg * x[:, 0] ** 2 * x[:, 7])
                + x[:, 2] / x[:, 4]
            )
        )
    )


# ---------------------------------------------------------------------------
# Space-filling criteria: deterministic
# ---------------------------------------------------------------------------


class TestCriteria:
    ref = REF["criteria"]
    X = np.array(REF["criteria"]["X"])

    def test_criteria_of_a_given_design(self):
        crit = sp.design_criteria(self.X)
        # same formula on the same eight runs: rounding error only
        assert crit["maxpro"] == pytest.approx(self.ref["maxpro"], rel=1e-12)
        assert crit["maximin"] == pytest.approx(self.ref["maximin"], rel=1e-12)
        assert crit["reciprocal_distance"] == pytest.approx(
            self.ref["reciprocal"], rel=1e-12
        )
        assert crit["wraparound_discrepancy"] == pytest.approx(
            self.ref["wraparound"], rel=1e-12
        )

    def test_options(self):
        assert sp.design_criteria(self.X, delta=1e-3)["maxpro"] == pytest.approx(
            self.ref["maxpro_delta"], rel=1e-12
        )
        assert sp.design_criteria(self.X, r=4)["reciprocal_distance"] == pytest.approx(
            self.ref["reciprocal_r4"], rel=1e-12
        )

    def test_energy_distance_to_a_sample(self):
        """``twinning::energy``: both arguments scaled by the sample."""
        from statspai.doe.criteria import energy_to_sample

        S = np.array(self.ref["S"])
        assert energy_to_sample(S, S[:25]) == pytest.approx(
            self.ref["energy"], rel=1e-9
        )

    def test_greedy_maxpro_augmentation_picks_the_same_runs(self):
        out = sp.design_augment(self.X, 6, candidates=np.array(self.ref["C"]))
        np.testing.assert_allclose(
            out.unit, np.array(self.ref["augmented"]), atol=1e-14
        )


# ---------------------------------------------------------------------------
# Searched designs: a stochastic screen, not parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,key,n,p,smaller_is_better",
    [
        ("maxpro", "maxpro", 20, 2, True),
        ("maximin", "maximin", 20, 2, False),
        ("uniform", "wraparound_discrepancy", 20, 2, True),
    ],
)
def test_searched_designs_are_on_par(method, key, n, p, smaller_is_better):
    """Our best of three seeds against the median of five SFDesign runs.

    The band is 3%: the spread of SFDesign's own five runs is 1 to 2% for
    MaxPro and the discrepancy and larger for maximin.
    """
    ref = REF["quality"][f"{method}_{n}_{p}"][1]
    ours = [
        sp.space_filling(n, p, method=method, seed=s).criteria[key] for s in range(3)
    ]
    if smaller_is_better:
        assert min(ours) <= ref * 1.03
    else:
        assert max(ours) >= ref * 0.97


# ---------------------------------------------------------------------------
# Fractional factorials: exact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(REF["frf2"]))
def test_minimum_aberration_fraction_has_frf2_word_length_pattern(key):
    runs, k = (int(v) for v in key.split("_"))
    d = sp.factorial_design(k, n_runs=runs)
    ref = REF["frf2"][key]
    assert d.resolution == ref["resolution"]
    # integers: the patterns are equal, not close
    assert d.word_length_pattern == [int(round(v)) for v in ref["gwlp"]]


class TestGeneralizedWordLength:
    ref = REF["gwlp"]

    def _a(self, design, max_order=None):
        s = sp.design_aberration(pd.DataFrame(np.array(design)), max_order=max_order)
        return [s[c] for c in s.index if c.startswith("A")]

    def test_replicated_l9(self):
        L9 = np.array(self.ref["L9"])
        np.testing.assert_allclose(
            self._a(np.vstack([L9, L9])), self.ref["L9_twice"], atol=1e-10
        )

    def test_columns_of_l18(self):
        L18 = np.array(self.ref["L18"])
        np.testing.assert_allclose(
            self._a(L18[:, 1:5]), self.ref["L18_cols_2_5"], atol=1e-10
        )
        np.testing.assert_allclose(
            self._a(L18[:, 2:6]), self.ref["L18_cols_3_6"], atol=1e-10
        )
        np.testing.assert_allclose(self._a(L18, 4), self.ref["L18_all_k4"], atol=1e-10)

    def test_unbalanced_mixed_level_design(self):
        np.testing.assert_allclose(
            self._a(self.ref["mixed"]), self.ref["mixed_gwlp"], atol=1e-10
        )

    def test_regular_fraction_equals_its_word_length_pattern(self):
        d = sp.factorial_design(7, n_runs=16)
        a = sp.design_aberration(d.design, max_order=6)
        np.testing.assert_allclose(
            [a[f"A{j}"] for j in range(1, 7)], d.word_length_pattern[:6], atol=1e-10
        )


# ---------------------------------------------------------------------------
# Unreplicated factorials
# ---------------------------------------------------------------------------


class TestFactorialEffects:
    ref = REF["lenth"]
    data = pd.DataFrame(REF["lenth"]["data"])

    def test_effects_and_pseudo_standard_error(self):
        fit = sp.factorial_effects(self.data, "y")
        names = [n.replace(":", "") for n in self.ref["names"]]
        got = fit.effects.loc[names, "effect"].to_numpy()
        np.testing.assert_allclose(got, self.ref["effects"], atol=1e-10)
        assert fit.pse == pytest.approx(self.ref["pse"], rel=1e-12)
        assert fit.df_resid == 0

    def test_pse_of_fifteen_effects(self):
        from statspai.doe.effects import lenth_pse

        assert lenth_pse(np.array(self.ref["e15"])) == pytest.approx(
            self.ref["pse15"], rel=1e-12
        )

    def test_simulated_margins_agree_with_unrepx_within_monte_carlo_error(self):
        """Both sides simulate the null; unrepx with about 40,000 effects.

        A screen: 3% on the margin of error (a 95% quantile of 40,000
        draws), 8% on the simultaneous one (a 95% quantile of 2,667
        maxima on the unrepx side).
        """
        fit = sp.factorial_effects(self.data, "y")
        me, sme = self.ref["me"]
        assert fit.me == pytest.approx(me, rel=0.03)
        assert fit.sme == pytest.approx(sme, rel=0.08)

    def test_t_reference_is_lenth_1989(self):
        from scipy import stats

        fit = sp.factorial_effects(self.data, "y", reference="t")
        m = fit.effects.shape[0]
        assert fit.me == pytest.approx(stats.t.ppf(0.975, m / 3) * fit.pse, rel=1e-12)
        gamma = 0.5 * (1 + 0.95 ** (1 / m))
        assert fit.sme == pytest.approx(stats.t.ppf(gamma, m / 3) * fit.pse, rel=1e-12)

    def test_half_fraction_reports_aliases(self):
        rows = np.array(self.ref["half"]["rows"]) - 1
        half = self.data.iloc[rows]
        fit = sp.factorial_effects(half, "y")
        # lm(y ~ (A + B + C)^3): the seven estimable columns
        ref = dict(
            zip(["A", "B", "C", "AB", "AC", "BC", "ABC"], self.ref["half"]["coef"][1:])
        )
        for term in ("A", "B", "C", "AB", "AC"):
            assert fit.effects.loc[term, "coef"] == pytest.approx(ref[term], abs=1e-10)
        # D = ABC in this half: terms are entered by order, so D and AD
        # stay and carry what R reports under ABC and BC
        assert fit.effects.loc["D", "coef"] == pytest.approx(ref["ABC"], abs=1e-10)
        assert fit.effects.loc["AD", "coef"] == pytest.approx(ref["BC"], abs=1e-10)
        assert "ABC" in fit.aliases["D"] and "BC" in fit.aliases["AD"]
        assert fit.aliases["(Intercept)"] == ["ABCD"]
        assert fit.effects.shape[0] == 7


# ---------------------------------------------------------------------------
# Support points, SPlit, twinning
# ---------------------------------------------------------------------------


class TestRepresentativePoints:
    ref = REF["support"]
    D = np.array(REF["support"]["D"])
    frame = pd.DataFrame(np.array(REF["support"]["D"]), columns=["a", "b"])

    @pytest.mark.parametrize(
        "key,r,u1",
        [("twin_r5_u1", 5, 1), ("twin_r4_u77", 4, 77), ("twin_r10_u300", 10, 300)],
    )
    def test_twinning_returns_the_rows_of_r(self, key, r, u1):
        split = sp.split_data(
            self.frame, test_size=1 / r, method="twinning", start=u1 - 1
        )
        assert np.array_equal(split.test_index + 1, np.sort(self.ref[key]))

    def test_twinning_with_a_factor_column(self):
        F = pd.DataFrame(self.ref["F"])
        split = sp.split_data(F, test_size=0.25, method="twinning", start=9)
        assert np.array_equal(split.test_index + 1, np.sort(self.ref["twin_F"]))

    def test_twinning_when_rows_do_not_divide(self):
        G = pd.DataFrame(np.array(self.ref["G"]), columns=list("abc"))
        idx = twin_indices(helmert_frame(G, list("abc")), 5, 2)
        assert np.array_equal(np.sort(idx) + 1, np.sort(self.ref["twin_G"]))

    def test_nearest_row_subsampling(self):
        """``SPlit::subsample`` on raw units: each point takes the nearest
        row not yet taken, in the order of the points."""
        got = _nearest_distinct(self.D, np.array(self.ref["SP"]))
        assert np.array_equal(got + 1, np.array(self.ref["subsample"]))

    def test_support_points_are_on_par(self):
        """A screen. Energy distances (``twinning::energy``) of five R runs
        lie within 3e-5 of each other; ours must not be worse than their
        worst by more than that."""
        from statspai.doe.criteria import energy_to_sample

        ours = [
            energy_to_sample(
                self.D, sp.support_points(self.frame, 40, seed=s).points.to_numpy()
            )
            for s in range(3)
        ]
        assert min(ours) <= max(self.ref["sp_energy"]) + 3e-5

    def test_split_is_on_par(self):
        """A screen, as above, for the 20% test set of SPlit."""
        ours = [
            sp.split_data(self.frame, 0.2, method="support", seed=s).energy
            for s in range(3)
        ]
        assert min(ours) <= max(self.ref["split_energy"]) + 3e-5


# ---------------------------------------------------------------------------
# Sensitivity analysis: deterministic given the runs
# ---------------------------------------------------------------------------


class TestSensitivity:
    ref = REF["sensitivity"]

    def test_jansen_estimates_after_the_divisor_rescaling(self):
        A, B = np.array(self.ref["A"]), np.array(self.ref["B"])
        res = sp.sobol_indices(
            borehole, 8, A=A, B=B, pass_as="array", n_boot=0, estimator="jansen"
        )
        n = A.shape[0]
        # sensitivity::soboljansen divides the sums of squares by 2n - 1,
        # the estimator of Jansen (1999) by 2n. Undo that and compare.
        scale = 2 * n / (2 * n - 1)
        np.testing.assert_allclose(
            res.indices["total"] * scale, self.ref["total"], atol=1e-12
        )
        np.testing.assert_allclose(
            1 - (1 - res.indices["first"]) * scale, self.ref["first"], atol=1e-12
        )

    def test_morris_statistics_from_the_same_trajectories(self):
        res = sp.morris_screening(
            None, 8, design=np.array(self.ref["morris_X"]), y=self.ref["morris_y"]
        )
        # elementary effects are of order 100
        np.testing.assert_allclose(res.effects["mu"], self.ref["mu"], atol=1e-10)
        np.testing.assert_allclose(
            res.effects["mu_star"], self.ref["mu_star"], atol=1e-10
        )
        np.testing.assert_allclose(res.effects["sigma"], self.ref["sigma"], atol=1e-10)


# ---------------------------------------------------------------------------
# Optimal designs: same objective, two optimisers
# ---------------------------------------------------------------------------


class TestOptimalDesigns:
    ref = REF["optimal"]

    def test_saturated_polynomial_design_equals_optfederov(self):
        formula = " + ".join(f"I(x**{j})" for j in range(1, 10))
        cand = pd.DataFrame({"x": np.linspace(-1, 1, 301)})
        d = sp.doe_optimal(formula, candidates=cand, n=10, seed=1)
        np.testing.assert_allclose(
            d.to_frame()["x"].to_numpy(), self.ref["poly9_x"], atol=1e-12
        )
        # D = det(X'X / n)^(1/k), about 4e-3: same design, same value
        assert d.criterion_value == pytest.approx(self.ref["poly9_D"], rel=1e-9)

    def test_three_factor_quadratic_reaches_optfederov_value(self):
        """14 runs from a 5^3 grid for a full quadratic. The problem has
        many local optima; with 20 starts each side finds the same value.
        Seeds are fixed, so this is a regression test of the search as
        much as a comparison."""
        terms = "a + b + c + I(a**2) + I(b**2) + I(c**2) + a:b + a:c + b:c"
        cand = pd.DataFrame(
            list(itertools.product(np.linspace(-1, 1, 5), repeat=3)),
            columns=list("abc"),
        )
        d = sp.doe_optimal(terms, candidates=cand, n=14, criterion="D", seed=1)
        assert d.criterion_value == pytest.approx(self.ref["quad3_D"], rel=1e-9)
        a = sp.doe_optimal(terms, candidates=cand, n=14, criterion="A", seed=1)
        assert a.criterion_value == pytest.approx(self.ref["quad3_A"], rel=1e-9)
        i = sp.doe_optimal(terms, candidates=cand, n=14, criterion="I", seed=1)
        # the average prediction variance of AlgDesign's design, computed
        # here from its runs; ours, optimised for it, must not exceed it
        from patsy import dmatrix

        ref_design = pd.DataFrame(self.ref["quad3_design"])
        G = np.asarray(dmatrix(terms, cand))
        X = np.asarray(dmatrix(terms, ref_design))
        i_ref = float(np.trace((G.T @ G / len(G)) @ np.linalg.inv(X.T @ X / len(X))))
        assert i.criterion_value <= i_ref * (1 + 1e-9)


# ---------------------------------------------------------------------------
# Kriging (rkriging)
# ---------------------------------------------------------------------------


class TestKriging:
    ref = REF["kriging"]
    df = pd.DataFrame({"x": REF["kriging"]["x"], "y": REF["kriging"]["y"]})
    test = pd.DataFrame({"x": REF["kriging"]["test"]})

    def test_prediction_at_given_hyperparameters(self):
        """Ordinary kriging with a Gaussian kernel. rkriging adds a nugget
        of 1e-6 (relative to the process variance) when it interpolates;
        with the same nugget the predictions agree to rounding."""
        fx = self.ref["fixed"]
        fit = sp.gp_regress(
            "y ~ x",
            self.df,
            length_scale=fx["lengthscale"],
            signal_var=fx["nu2"],
            noise_var=1e-6 * fx["nu2"],
            optimize_hyper=False,
        )
        pred = fit.predict(self.test)
        assert fit.params["mean"] == pytest.approx(fx["mu"], abs=1e-10)
        np.testing.assert_allclose(pred["mean"], fx["mean"], atol=1e-10)
        np.testing.assert_allclose(pred["sd"], fx["sd"], atol=1e-10)

    def test_fitted_interpolator(self):
        """Both maximise the restricted likelihood. 1e-4 relative on the
        hyperparameters is the agreement of two optimisers; the nugget
        differs (1e-8 against 1e-6), which moves predictions by 1e-5."""
        ft = self.ref["fitted"]
        fit = sp.gp_regress("y ~ x", self.df, interpolate=True, seed=0)
        assert fit.params["length_scale[x]"] == pytest.approx(
            ft["lengthscale"], rel=1e-4
        )
        assert fit.params["signal_var"] == pytest.approx(ft["nu2"], rel=1e-4)
        pred = fit.predict(self.test)
        np.testing.assert_allclose(pred["mean"], ft["mean"], atol=2e-4)
        np.testing.assert_allclose(pred["sd"], ft["sd"], atol=2e-4)
        ei = fit.expected_improvement(self.test)
        np.testing.assert_allclose(ei, ft["ei"], atol=2e-4)
        assert int(ei.to_numpy().argmax()) == int(np.argmax(ft["ei"]))

    def test_noisy_replicated_design_finds_the_smooth_mode(self):
        """Ten sites observed twice. The likelihood has a second, lower
        mode at a length scale far below the spacing of the sites, where
        the surface is flat; before 1.39 the optimiser stopped there and
        the prediction between sites collapsed to the mean (a difference
        of 0.79 from the reference on a function of range 1.5)."""
        nz = self.ref["noisy"]
        d = pd.DataFrame({"x": nz["x"], "y": nz["y"]})
        fit = sp.gp_regress("y ~ x", d, seed=0)
        assert fit.params["length_scale[x]"] == pytest.approx(
            nz["lengthscale"], rel=1e-3
        )
        assert fit.params["noise_var"] == pytest.approx(nz["sigma2"], rel=1e-3)
        pred = fit.predict(self.test)
        np.testing.assert_allclose(pred["mean"], nz["mean"], atol=1e-4)
        np.testing.assert_allclose(pred["sd"], nz["sd"], atol=1e-4)


# ---------------------------------------------------------------------------
# Second round: qualitative factors, larger fractions, factor importance
# ---------------------------------------------------------------------------


class TestQualitativeFactors:
    ref = REF["qualitative"]

    def _criterion(self, groups):
        """The mixed-factor MaxPro criterion, written out: each pair of
        runs contributes 1 / prod (x_il - x_jl)^2 / prod (d_k + 1/L_k)^2
        with d_k = 1 when the runs differ in qualitative factor k."""
        L = np.array(self.ref["L"])
        n = len(L)
        iu = np.triu_indices(n, k=1)
        lg = np.zeros(iu[0].size)
        for col in L.T:
            lg -= 2 * np.log(np.abs(col[iu[0]] - col[iu[1]]))
        for g in groups:
            g = np.array(g)
            d = (g[iu[0]] != g[iu[1]]).astype(float)
            lg -= 2 * np.log(d + 1 / len(set(g)))
        return np.exp(lg).mean() ** (1 / (L.shape[1] + len(groups)))

    def test_criterion_formula_is_that_of_maxpromeasure(self):
        r = self.ref
        assert self._criterion([r["g1"]]) == pytest.approx(r["one"], rel=1e-12)
        assert self._criterion([r["g2"]]) == pytest.approx(r["other"], rel=1e-12)
        assert self._criterion([r["g1"], r["g2"]]) == pytest.approx(
            r["both"], rel=1e-12
        )

    def test_reported_criterion_is_that_formula(self):
        d = sp.space_filling(18, 2, seed=1, qualitative={"g": [1, 2, 3]})
        X = d.unit
        g = d.design["g"].to_numpy()
        iu = np.triu_indices(18, k=1)
        lg = np.zeros(iu[0].size)
        for col in X.T:
            lg -= 2 * np.log(np.abs(col[iu[0]] - col[iu[1]]))
        lg -= 2 * np.log((g[iu[0]] != g[iu[1]]).astype(float) + 1 / 3)
        assert d.criteria["maxpro_qq"] == pytest.approx(
            np.exp(lg).mean() ** (1 / 3), rel=1e-10
        )

    def test_searched_design_is_on_par(self):
        """A screen. MaxProQQ permutes the levels of a Latin hypercube;
        ours also moves them continuously, so it should not be worse."""
        ours = min(
            sp.space_filling(18, 2, seed=s, qualitative={"g": [1, 2, 3]}).criteria[
                "maxpro_qq"
            ]
            for s in range(2)
        )
        assert ours <= min(self.ref["searched_18_2_3"]) * 1.02


@pytest.mark.parametrize("key", sorted(REF["frf2_large"]))
def test_larger_minimum_aberration_searches(key):
    """The compiled exhaustive search against the FrF2 catalogue."""
    runs, k = (int(v) for v in key.split("_"))
    d = sp.factorial_design(k, n_runs=runs)
    ref = REF["frf2_large"][key]
    assert d.resolution == ref["resolution"]
    assert d.word_length_pattern[:6] == [int(round(v)) for v in ref["gwlp"][:6]]


class TestOtherFractions:
    ref = REF["arrays"]

    def test_l18_has_the_pattern_of_the_catalogued_array(self):
        """Built from a difference matrix found by search; an orthogonal
        array is defined up to isomorphism, so the generalized word
        length pattern is what can be compared."""
        d = sp.factorial_design(8, levels=[2] + [3] * 7, n_runs=18)
        np.testing.assert_allclose(
            d.word_length_pattern[:5], self.ref["L18_gwlp"][:5], atol=1e-8
        )

    def test_l9(self):
        d = sp.factorial_design(4, levels=3, n_runs=9)
        np.testing.assert_allclose(
            d.word_length_pattern[:4], self.ref["L9_gwlp"][:4], atol=1e-8
        )

    def test_five_three_level_factors_in_27_runs_is_no_worse(self):
        """DoE.base picks columns of its L27; ours searches for minimum
        aberration, so its pattern must be at most the catalogue's."""
        d = sp.factorial_design(5, levels=3, n_runs=27)
        ours = [round(v, 8) for v in d.word_length_pattern[:5]]
        ref = [round(v, 8) for v in self.ref["L27_5"][:5]]
        assert ours <= ref


class TestFactorImportance:
    ref = REF["first"]

    def _frame(self):
        df = pd.DataFrame(np.array(self.ref["D"]), columns=list("abcd"))
        df["y"] = self.ref["y"]
        return df

    @pytest.mark.parametrize(
        "kwargs,key",
        [
            ({}, "default"),
            ({"n_forward": 4}, "forward4"),
            ({"n_neighbors": 5}, "knn5"),
            ({"factors": ["a", "c", "d"]}, "subset"),
            ({"standardize": False}, "raw"),
        ],
    )
    def test_equals_first(self, kwargs, key):
        """Deterministic given the data: nearest neighbours, sample
        variances and a greedy selection. Rounding error only."""
        res = sp.factor_importance(self._frame(), "y", **kwargs)
        np.testing.assert_allclose(res.importance, self.ref[key], atol=1e-10)

    def test_ishigami_with_noise(self):
        df = pd.DataFrame(np.array(self.ref["Xi"]), columns=[f"x{i}" for i in range(6)])
        df["y"] = self.ref["yi"]
        res = sp.factor_importance(df, "y", standardize=False)
        np.testing.assert_allclose(res.importance, self.ref["ishigami"], atol=1e-10)

    def test_binary_outcome(self):
        df = self._frame().drop(columns="y")
        df["y"] = self.ref["yb"]
        res = sp.factor_importance(df, "y")
        np.testing.assert_allclose(res.importance, self.ref["binary"], atol=1e-10)

    def test_categorical_factor(self):
        """y = 2 a + 3 [g = v] + noise. The importance of ``g`` needs the
        conditional variance given ``a`` and equals the reference. That
        of ``a`` needs the conditional variance given ``g`` alone, where
        every observation of a level is a tied nearest neighbour: the
        reference takes whichever the tree returns (0.549), we take the
        variance within the level. The population value is 4 / 6."""
        F = pd.DataFrame(self.ref["F"])
        F["y"] = self.ref["yg"]
        res = sp.factor_importance(F, "y")
        assert res.importance["g"] == pytest.approx(self.ref["factor"][1], abs=1e-10)
        assert res.importance["c"] == 0.0
        assert self.ref["factor"][0] == pytest.approx(0.549, abs=1e-3)
        # 0.619 in this sample of 400
        assert res.importance["a"] == pytest.approx(4 / 6, abs=0.06)
        assert abs(res.importance["a"] - 4 / 6) < abs(self.ref["factor"][0] - 4 / 6)
