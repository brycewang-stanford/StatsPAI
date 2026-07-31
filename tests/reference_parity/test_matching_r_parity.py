"""Cross-package parity for the matching / weighting module.

Every reference number lives in ``_fixtures/matching_R.json``, produced by
``_fixtures/_generate_matching_r.R`` from CBPS 0.24, ebal 0.2.1,
MatchIt 4.7.2 and optmatch 0.10.8 on ``_fixtures/matching_lalonde.csv``
(``MatchIt::lalonde``, 614 obs / 185 treated). R is not executed in CI.

Three kinds of assertion appear here, and the distinction matters:

1. **Parity** — StatsPAI must reproduce the R number to a stated tolerance.
2. **Dominance** — where the R implementation stops short of its own
   optimum, we assert StatsPAI is *at least as good* on the objective the
   estimator is defined by (exact covariate balance, GMM loss, total
   matched distance) rather than pinning to R's under-converged value.
   Each such case is justified in the test's docstring.
3. **Self-consistency** — properties that follow from the estimator's
   definition and need no reference at all (e.g. just-identified CBPS
   balances exactly).

Reference tolerances are deliberately *not* uniform: they encode how far
each reference implementation is from its own stationary point.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIXTURES = Path(__file__).parent / "_fixtures"
COV = ["age", "educ", "married", "nodegree", "re74", "re75", "black", "hispan"]


@pytest.fixture(scope="module")
def R():
    return json.loads((FIXTURES / "matching_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lalonde():
    return pd.read_csv(FIXTURES / "matching_lalonde.csv")


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-12)


# ===================================================================== #
#  Logistic propensity score — the shared anchor
# ===================================================================== #


def test_logit_propensity_matches_r_glm(lalonde, R):
    """Everything downstream conditions on this score, so pin it first."""
    import statsmodels.api as sm

    X = sm.add_constant(lalonde[COV].to_numpy(float))
    fit = sm.Logit(lalonde["treat"].to_numpy(float), X).fit(disp=False)
    assert np.allclose(fit.params, R["logit_ps"]["coefficients"], rtol=1e-6)


# ===================================================================== #
#  CBPS — Imai & Ratkovic (2014)
# ===================================================================== #


class TestCBPS:
    """``sp.cbps`` vs ``CBPS::CBPS``.

    ATE (both variants) and the just-identified ATT agree with R. The
    over-identified ATT deliberately does not: see
    ``test_cbps_att_over_dominates_r``.
    """

    @pytest.mark.parametrize(
        "estimand,variant,rtol",
        [
            # Just-identified: both sides solve the same K equations in K
            # unknowns, so the only gap is R's optimiser slack (CBPS stops
            # around 1e-10 on the balance loss; StatsPAI reaches ~1e-20).
            ("ATT", "exact", 2e-3),
            ("ATE", "exact", 2e-3),
            # Over-identified ATE: same frozen weighting matrix, same
            # optimum, agreement limited by BFGS termination on both sides.
            ("ATE", "over", 5e-3),
        ],
    )
    def test_cbps_matches_r(self, lalonde, R, estimand, variant, rtol):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.cbps(
                lalonde,
                y="re78",
                treat="treat",
                covariates=COV,
                estimand=estimand,
                variant=variant,
                n_bootstrap=2,
                seed=0,
            )
        want = R[f"cbps_{estimand.lower()}_{variant}"]["att"]
        assert _rel(float(res.estimate), want) < rtol, (
            f"CBPS {estimand}/{variant}: {float(res.estimate):.6f} vs R "
            f"{want:.6f} (rel {_rel(float(res.estimate), want):.2e})"
        )

    @pytest.mark.parametrize("estimand", ["ATT", "ATE"])
    def test_cbps_exact_balances_exactly(self, lalonde, estimand):
        """Self-consistency: the just-identified variant has K balance
        equations in K unknowns, so the weighted covariate means must equal
        their targets to solver precision. No reference needed."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.cbps(
                lalonde,
                y="re78",
                treat="treat",
                covariates=COV,
                estimand=estimand,
                variant="exact",
                n_bootstrap=2,
                seed=0,
            )
        smd = res.model_info["std_mean_diff_after"]
        worst = max(abs(v) for k, v in smd.items() if k != "_intercept")
        # 1e-6 rather than machine zero: the ATT balance moments carry a
        # n/n_1 factor (~3.3 here) that amplifies the dual residual. R's
        # CBPS leaves ~1e-3 on the same quantity.
        assert worst < 1e-6, f"just-identified CBPS left |SMD| = {worst:.3e}"

    def test_cbps_att_over_dominates_r(self, lalonde, R):
        """Over-identified ATT: StatsPAI must balance strictly better than R.

        CBPS's analytic ATT gradient divides the balance block by ``n_1``
        where the moment's Jacobian carries ``1/n``, overstating that block
        by ``n/n_1``; its ``optim`` call consequently stops at a
        non-stationary point. StatsPAI uses the correct Jacobian, so pinning
        to R's value would be pinning to R's bug. What we can assert is the
        thing CBPS is *for*: covariate balance under the fitted weights.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.cbps(
                lalonde,
                y="re78",
                treat="treat",
                covariates=COV,
                estimand="ATT",
                variant="over",
                n_bootstrap=2,
                seed=0,
            )
        X = lalonde[COV].to_numpy(float)
        T = lalonde["treat"].to_numpy(float)
        Xd = np.column_stack([np.ones(len(lalonde)), X])

        def worst_smd(beta):
            ps = np.clip(
                1 / (1 + np.exp(-(Xd @ np.asarray(beta, float)))), 1e-6, 1 - 1e-6
            )
            w = np.where(T == 1, 1.0, ps / (1 - ps))
            mt = X[T == 1].mean(0)
            mc = (X[T == 0] * w[T == 0, None]).sum(0) / w[T == 0].sum()
            return float(np.max(np.abs(mt - mc) / X[T == 1].std(0, ddof=1)))

        ours = worst_smd(res.model_info["beta"])
        theirs = worst_smd(R["cbps_att_over"]["coefficients"])
        assert ours < theirs, (
            f"StatsPAI CBPS ATT/over max |SMD| {ours:.4f} should beat "
            f"CBPS::CBPS {theirs:.4f}"
        )
        # Guard the margin so a regression that merely ties cannot pass.
        assert ours < 0.05 and theirs > 0.10


# ===================================================================== #
#  Entropy balancing — Hainmueller (2012)
# ===================================================================== #


class TestEntropyBalancing:
    def test_ebalance_matches_r(self, lalonde, R):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.ebalance(lalonde, y="re78", treat="treat", covariates=COV)
        want = R["ebal_att"]["att"]
        assert (
            _rel(float(res.estimate), want) < 1e-5
        ), f"ebalance ATT {float(res.estimate):.6f} vs ebal {want:.6f}"

    def test_ebalance_balance_is_at_least_as_exact_as_ebal(self, lalonde, R):
        """Entropy balancing is *defined* by exact moment matching, so the
        achieved gap is the estimator's own correctness criterion. StatsPAI
        solves the dual to a true stationary point (gap ~1e-15 relative);
        ``ebal`` stops around 1e-7."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.ebalance(lalonde, y="re78", treat="treat", covariates=COV)
        X = lalonde[COV].to_numpy(float)
        T = lalonde["treat"].to_numpy(float)
        w = np.asarray(res.model_info["weights"], float)
        mt = X[T == 1].mean(0)
        mc = (X[T == 0] * w[:, None]).sum(0) / w.sum()
        ours = float(np.max(np.abs(mt - mc) / np.maximum(np.abs(mt), 1e-12)))

        r_t = np.asarray(R["ebal_att"]["bal_treated"], float)
        r_c = np.asarray(R["ebal_att"]["bal_control_weighted"], float)
        theirs = float(np.max(np.abs(r_t - r_c) / np.maximum(np.abs(r_t), 1e-12)))

        assert ours < 1e-10, f"entropy balancing left relative gap {ours:.2e}"
        assert ours <= theirs

    def test_weights_full_aligns_with_input_rows(self, lalonde):
        """`weights` is control-only (the ebal convention); `weights_full`
        is the join-able version. Regression guard on the two shapes."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.ebalance(lalonde, y="re78", treat="treat", covariates=COV)
        T = lalonde["treat"].to_numpy()
        assert len(res.model_info["weights"]) == int((T == 0).sum())
        wf = np.asarray(res.model_info["weights_full"], float)
        assert len(wf) == len(lalonde)
        assert np.allclose(wf[T == 1], 1.0)


# ===================================================================== #
#  Nearest-neighbour matching vs MatchIt
# ===================================================================== #


class TestNearestNeighbourMatchIt:
    # ratio 3 is deliberately excluded: 185 treated x 3 exceeds the 429
    # available controls, so MatchIt exhausts the pool and falls back to
    # its own rule for treated units that receive fewer than k matches.
    # That fallback -- not the matching -- is what would be under test.
    @pytest.mark.parametrize("ratio", [1, 2])
    def test_ps_nn_without_replacement_matches_matchit(self, lalonde, R, ratio):
        """k:1 propensity-score matching without replacement, MatchIt's
        best-understood configuration. Exact to floating point."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.match(
                lalonde,
                y="re78",
                treat="treat",
                covariates=COV,
                distance="propensity",
                method="nearest",
                estimand="ATT",
                replace=False,
                n_matches=ratio,
            )
        want = R[f"matchit_nn_ps_noreplace_{ratio}"]["att"]
        assert (
            _rel(float(res.estimate), want) < 1e-9
        ), f"{ratio}:1 NN ATT {float(res.estimate):.9f} vs MatchIt {want:.9f}"


class TestMahalanobisMetric:
    """The Mahalanobis metric itself, isolated from assignment heuristics.

    ``sp.match(mahalanobis_cov='pooled')`` (the default since the metric was
    corrected) reproduces ``MatchIt:::mahalanobis_dist`` exactly. The
    previous default used the *total* covariance, which is inflated along
    the direction the group means differ in.
    """

    @staticmethod
    def _dist(lalonde, cov_kind):
        X = lalonde[COV].to_numpy(float)
        T = lalonde["treat"].to_numpy(int)
        it, ic = np.flatnonzero(T == 1), np.flatnonzero(T == 0)
        n1, n0 = len(it), len(ic)
        if cov_kind == "pooled":
            s1, s0 = np.cov(X[it].T), np.cov(X[ic].T)
            S = ((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2)
        else:
            S = np.cov(X.T)
        VI = np.linalg.inv(S)
        D = np.empty((n1, n0))
        for a in range(n1):
            d = X[ic] - X[it[a]]
            D[a] = np.sqrt(np.einsum("ij,jk,ik->i", d, VI, d))
        return D

    def test_pooled_covariance_reproduces_matchit_distance(self, lalonde, R):
        D = self._dist(lalonde, "pooled")
        ref = R["mahalanobis_dist"]
        assert _rel(D[0, 0], ref["d_00"]) < 1e-10
        assert _rel(D[0, 1], ref["d_01"]) < 1e-10
        assert _rel(D[1, 0], ref["d_10"]) < 1e-10
        assert _rel(float(np.sqrt((D**2).sum())), ref["frobenius"]) < 1e-10
        assert int(np.argmin(D[0])) == ref["row0_argmin"]

    def test_total_covariance_is_a_different_metric(self, lalonde, R):
        """Guard the correctness fix: the legacy metric must NOT match."""
        D = self._dist(lalonde, "total")
        assert _rel(D[0, 0], R["mahalanobis_dist"]["d_00"]) > 1e-3


class TestGreedyOrder:
    """Greedy NN without replacement is order-dependent, and StatsPAI's
    ``m_order`` values reproduce MatchIt's rules exactly when both run on
    the same supplied distance matrix.

    On lalonde the choice moves the Mahalanobis ATT across a >5x range, so
    this is a substantive user-facing knob rather than a tie-break detail.
    """

    # 'data' and 'closest' reproduce MatchIt exactly. 'farthest' is NOT
    # asserted against MatchIt: MatchIt's rule of that name does not agree
    # with the natural dynamic reading (pick the treated unit whose nearest
    # available control is furthest), and we did not reverse-engineer what
    # it does instead. StatsPAI's 'farthest' is the dynamic rule, and is
    # documented as such rather than claimed as MatchIt parity.
    @pytest.mark.parametrize("m_order", ["data", "closest"])
    def test_m_order_matches_matchit_on_supplied_distance(self, lalonde, R, m_order):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.match(
                lalonde,
                y="re78",
                treat="treat",
                covariates=COV,
                distance="mahalanobis",
                method="nearest",
                estimand="ATT",
                replace=False,
                m_order=m_order,
                mahalanobis_cov="pooled",
            )
        want = R[f"greedy_supplied_D_{m_order}"]["att"]
        assert _rel(float(res.estimate), want) < 1e-9, (
            f"m_order={m_order}: {float(res.estimate):.6f} vs MatchIt " f"{want:.6f}"
        )

    def test_order_choice_is_material(self, lalonde):
        """If this ever collapses, the parameter has silently stopped
        working — the spread is the whole reason it is exposed."""
        outs = []
        for mo in ("data", "closest", "farthest"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outs.append(
                    float(
                        sp.match(
                            lalonde,
                            y="re78",
                            treat="treat",
                            covariates=COV,
                            distance="mahalanobis",
                            method="nearest",
                            estimand="ATT",
                            replace=False,
                            m_order=mo,
                        ).estimate
                    )
                )
        assert max(outs) / min(outs) > 2.0


class TestCaliperScale:
    def test_sd_scale_uses_propensity_sd(self, lalonde, R):
        """``caliper_scale='sd'`` must equal a raw caliper of
        ``caliper * sd(propensity)`` — the MatchIt ``std.caliper = TRUE``
        convention, expressed against R's own sd(ps)."""
        sd_ps = R["logit_ps"]["sd_ps"]
        kw = dict(
            y="re78",
            treat="treat",
            covariates=COV,
            distance="propensity",
            method="nearest",
            estimand="ATT",
            replace=False,
            m_order="data",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = sp.match(lalonde, caliper=0.2, caliper_scale="sd", **kw)
            b = sp.match(lalonde, caliper=0.2 * sd_ps, caliper_scale="raw", **kw)
        assert _rel(float(a.estimate), float(b.estimate)) < 1e-12

    def test_raw_is_the_default(self, lalonde):
        """Default must stay the Stata psmatch2 (raw-units) convention."""
        kw = dict(
            y="re78",
            treat="treat",
            covariates=COV,
            distance="propensity",
            method="nearest",
            estimand="ATT",
            replace=False,
            caliper=0.05,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = sp.match(lalonde, **kw)
            b = sp.match(lalonde, caliper_scale="raw", **kw)
        assert float(a.estimate) == float(b.estimate)


# ===================================================================== #
#  Optimal matching vs optmatch
# ===================================================================== #


class TestOptimalMatching:
    def test_total_distance_is_at_least_as_low_as_optmatch(self, lalonde, R):
        """``sp.optimal_match`` solves the assignment problem exactly with
        the Hungarian algorithm; ``optmatch::pairmatch`` discretises
        distances for its network-flow solver. We require our objective to
        be no worse.

        The matched *pairs* need not agree: the assignment problem is
        degenerate here (many near-ties in the propensity score), so two
        equally optimal solutions can report different ATTs. That is why
        the objective, not the ATT, is what gets pinned.
        """
        res = sp.optimal_match(
            lalonde,
            treatment="treat",
            outcome="re78",
            covariates=COV,
            metric="propensity",
        )
        ours = float(res.distances.sum())
        theirs = R["optmatch_pair_ps"]["total_distance"]
        assert res.n_matched == R["optmatch_pair_ps"]["n_pairs"]
        assert ours <= theirs * (1 + 1e-6), (
            f"total matched distance {ours:.6f} worse than optmatch " f"{theirs:.6f}"
        )

    def test_att_alias(self, lalonde):
        res = sp.optimal_match(
            lalonde, treatment="treat", outcome="re78", covariates=COV
        )
        assert res.att == res.ate == res.estimate


# ===================================================================== #
#  Matching::Match — ties, Abadie-Imbens variance
# ===================================================================== #


class TestMatchingPackageParity:
    """``sp.match`` vs ``Matching::Match`` (Sekhon 2011) with replacement.

    ``Match`` pools every control whose squared inverse-variance-weighted
    distance is within ``distance.tolerance`` (1e-5) of the minimum, rather
    than breaking ties by row order, and reports the Abadie-Imbens
    *population* ATT variance. Both conventions are reproduced exactly by
    ``ties='all', tie_tolerance=1e-5`` and ``se_method='abadie_imbens_pop'``.
    """

    @staticmethod
    def _fit(lalonde, **kw):
        kw.setdefault("tie_tolerance", 1e-5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sp.match(
                lalonde,
                y="re78",
                treat="treat",
                covariates=COV,
                distance="propensity",
                method="nearest",
                estimand="ATT",
                replace=True,
                ties="all",
                **kw,
            )

    @pytest.mark.parametrize("m", [1, 3])
    def test_att_matches_matching_package(self, lalonde, R, m):
        res = self._fit(lalonde, n_matches=m)
        want = R[f"aimatch_M{m}_biasF"]["est"]
        assert (
            _rel(float(res.estimate), want) < 1e-9
        ), f"M={m}: {float(res.estimate):.6f} vs Match {want:.6f}"

    @pytest.mark.parametrize("m", [1, 3])
    def test_abadie_imbens_population_se_matches(self, lalonde, R, m):
        res = self._fit(lalonde, n_matches=m, se_method="abadie_imbens_pop")
        want = R[f"aimatch_M{m}_biasF"]["se"]
        assert (
            _rel(float(res.se), want) < 1e-8
        ), f"M={m}: SE {float(res.se):.6f} vs Match {want:.6f}"

    def test_tie_tolerance_is_load_bearing(self, lalonde, R):
        """Not a cosmetic option: dropping the tolerance changes the tie set
        and must move the estimate away from R."""
        want = R["aimatch_M1_biasF"]["est"]
        with_tol = float(self._fit(lalonde, n_matches=1).estimate)
        no_tol = float(self._fit(lalonde, n_matches=1, tie_tolerance=0.0).estimate)
        assert _rel(with_tol, want) < 1e-9
        assert _rel(no_tol, want) > 1e-3

    def test_ai_population_se_differs_from_stata_ai(self, lalonde):
        """`abadie_imbens` (Stata psmatch2 sample-ATT) and
        `abadie_imbens_pop` (Matching::Match population-ATT) are different
        estimands and must not silently coincide."""
        pop = float(self._fit(lalonde, se_method="abadie_imbens_pop").se)
        sample = float(self._fit(lalonde, se_method="abadie_imbens").se)
        assert pop > 0 and sample > 0
        assert _rel(pop, sample) > 0.02


# ===================================================================== #
#  Stable balancing weights vs sbw::sbw
# ===================================================================== #


class TestStableBalancingWeights:
    @pytest.mark.parametrize(
        "scale,tol,key",
        [
            ("target", 0.05, "sbw_target_005"),
            ("target", 0.02, "sbw_target_002"),
            ("group", 0.05, "sbw_group_005"),
            ("group", 0.02, "sbw_group_002"),
        ],
    )
    def test_sbw_matches_r(self, lalonde, R, scale, tol, key):
        """`tolerance_scale` names the standard deviation `delta` is quoted
        in: 'target' is sbw::sbw's bal_std="target" (treated sd) and 'group'
        is bal_std="group" (control sd). A tolerance quoted without its scale
        is not reproducible."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.sbw(
                lalonde,
                treat="treat",
                covariates=COV,
                y="re78",
                estimand="att",
                delta=tol,
                tolerance_scale=scale,
            )
        want = R[key]["att"]
        assert (
            _rel(float(res.estimate), want) < 1e-8
        ), f"sbw {scale}/{tol}: {float(res.estimate):.6f} vs {want:.6f}"

    def test_scale_choice_moves_the_estimate(self, lalonde):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = sp.sbw(
                lalonde,
                treat="treat",
                covariates=COV,
                y="re78",
                delta=0.05,
                tolerance_scale="target",
            )
            b = sp.sbw(
                lalonde,
                treat="treat",
                covariates=COV,
                y="re78",
                delta=0.05,
                tolerance_scale="group",
            )
        assert _rel(float(a.estimate), float(b.estimate)) > 1e-3


# ===================================================================== #
#  Genetic matching kernel
# ===================================================================== #


class TestGenMatchKernel:
    def test_weighted_distance_kernel_matches_matching_package(self, lalonde, R):
        """The genetic search is stochastic and cannot be reproduced across
        languages; the deterministic kernel it searches over can be. Given the
        same diagonal W, our 1-NN assignment must agree with
        ``Matching::Match(Weight = 3, Weight.matrix = W)`` on every treated
        unit R matched uniquely (ties are pooled by R and so are not a
        statement about the metric)."""
        from statspai.matching.genmatch import _match_with_weights

        ref = R["genmatch_weight_matrix"]
        X = lalonde[COV].to_numpy(float)
        T = lalonde["treat"].to_numpy(int)
        it = np.flatnonzero(T == 1)
        ic = np.flatnonzero(T == 0)
        w = np.asarray(ref["w_diag"], dtype=float)

        m = _match_with_weights(X[it], X[ic], w, 1)
        ours = {int(it[r]): int(ic[m[r, 0]]) for r in range(len(it))}

        want_t = np.atleast_1d(np.asarray(ref["unique_treated"], dtype=int))
        want_c = np.atleast_1d(np.asarray(ref["unique_control"], dtype=int))
        assert len(want_t) == ref["n_unique_treated"]
        mismatches = [
            (int(a), ours[int(a)], int(b))
            for a, b in zip(want_t, want_c)
            if ours[int(a)] != int(b)
        ]
        assert not mismatches, (
            f"{len(mismatches)}/{len(want_t)} uniquely-matched treated units "
            f"disagree with Matching::Match; first: {mismatches[:3]}"
        )
