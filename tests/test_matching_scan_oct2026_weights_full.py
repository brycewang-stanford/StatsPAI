"""ps_weights / ess / energy_distance / implied_weights, full matching,
sp.pscore, sp.cbps and two-criteria matching against hand computation.

Covers the branches of ``matching/ps_weights.py``, ``full.py``, ``pscore.py``,
``cbps.py`` and ``two_criteria.py`` the suite did not reach.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.spatial.distance import cdist

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.matching.full import full_match_sets, matched_set_effect
from statspai.matching.two_criteria import rank_mahalanobis

X = ["x1", "x2"]


def _data(seed: int = 0, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2))))
    y = 1 + 2 * d + x1 + 0.5 * x2 + d * x1 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


# ----------------------------------------------------------------------
# ps_weights, ess, energy_distance, implied_weights
# ----------------------------------------------------------------------


class TestPsWeights:
    def _et(self, n=50, seed=0):
        rng = np.random.default_rng(seed)
        e = rng.uniform(0.05, 0.95, n)
        return e, rng.binomial(1, e).astype(float)

    def test_every_estimand_from_its_tilting_function(self):
        e, t = self._et()
        tilt = {
            "ATE": np.ones_like(e),
            "ATT": e,
            "ATC": 1 - e,
            "ATO": e * (1 - e),
            "ATM": np.minimum(e, 1 - e),
        }
        for name, h in tilt.items():
            want = np.where(t == 1, h / e, h / (1 - e))
            np.testing.assert_allclose(sp.ps_weights(e, t, name), want, rtol=1e-14)
        # the treated carry weight one under the ATT, the controls under ATC
        assert (sp.ps_weights(e, t, "att")[t == 1] == 1).all()
        np.testing.assert_allclose(sp.ps_weights(e, t, "ATC")[t == 0], 1.0)

    def test_atc_is_the_att_of_the_relabelled_problem(self):
        e, t = self._et()
        np.testing.assert_allclose(
            sp.ps_weights(e, t, "ATC"), sp.ps_weights(1 - e, 1 - t, "ATT"), rtol=1e-14
        )
        np.testing.assert_allclose(
            sp.ps_weights(e, t, "ATU"), sp.ps_weights(e, t, "ATC"), rtol=0
        )

    def test_ipw_estimates_written_out(self):
        # the Hajek ATE with these weights is the textbook IPW estimator
        df = _data()
        e = sp.propensity_score(df, "d", X).to_numpy()
        t, y = df.d.to_numpy(dtype=float), df.y.to_numpy()
        w = sp.ps_weights(e, t)
        hajek = np.sum(w * t * y) / np.sum(w * t) - np.sum(w * (1 - t) * y) / np.sum(
            w * (1 - t)
        )
        direct = np.sum(t * y / e) / np.sum(t / e) - np.sum(
            (1 - t) * y / (1 - e)
        ) / np.sum((1 - t) / (1 - e))
        assert hajek == pytest.approx(direct, rel=1e-13)

    def test_stabilized_weights_keep_the_arm_means(self):
        e, t = self._et()
        w, sw = sp.ps_weights(e, t), sp.ps_weights(e, t, stabilize=True)
        p = t.mean()
        np.testing.assert_allclose(sw, w * np.where(t == 1, p, 1 - p), rtol=1e-14)

    def test_truncation_on_the_score_and_on_its_quantiles(self):
        e, t = self._et()
        clipped = np.clip(e, 0.2, 0.8)
        np.testing.assert_allclose(
            sp.ps_weights(e, t, truncate=(0.2, 0.8)),
            np.where(t == 1, 1 / clipped, 1 / (1 - clipped)),
            rtol=1e-14,
        )
        lo, hi = np.quantile(e, [0.1, 0.9])
        q = np.clip(e, lo, hi)
        np.testing.assert_allclose(
            sp.ps_weights(e, t, truncate=(0.1, 0.9), truncate_scale="quantile"),
            np.where(t == 1, 1 / q, 1 / (1 - q)),
            rtol=1e-14,
        )

    def test_a_series_in_gives_a_series_out(self):
        e, t = self._et()
        idx = pd.Index(np.arange(100, 150))
        out = sp.ps_weights(pd.Series(e, index=idx), t)
        assert isinstance(out, pd.Series) and out.index.equals(idx)
        assert out.name == "ps_weight"
        out2 = sp.ps_weights(e, pd.Series(t, index=idx))
        assert out2.index.equals(idx)
        assert isinstance(sp.ps_weights(e, t), np.ndarray)

    def test_continuous_exposure(self):
        rng = np.random.default_rng(1)
        mu = rng.normal(size=40)
        x = mu + rng.normal(scale=0.7, size=40)
        w = sp.ps_weights(mu, x, exposure="continuous", sigma=0.7)
        np.testing.assert_allclose(w, 1 / stats.norm.pdf(x, mu, 0.7), rtol=1e-14)
        sw = sp.ps_weights(mu, x, exposure="continuous", sigma=0.7, stabilize=True)
        np.testing.assert_allclose(
            sw, stats.norm.pdf(x, x.mean(), x.std(ddof=1)) * w, rtol=1e-14
        )

    @pytest.mark.parametrize(
        "kw, msg",
        [
            ({"ps": []}, "ps is empty"),
            ({"ps": [0.5, 0.5]}, "same length"),
            ({"ps": [0.5, np.nan, 0.5]}, "must not contain missing"),
            ({"estimand": "nope"}, "estimand must be"),
            ({"treat": [0, 1, 2]}, "treat must be 0/1"),
            ({"ps": [0.5, 1.0, 0.5]}, "strictly between 0 and 1"),
            ({"ps": [0.5, 0.0, 0.5]}, "strictly between 0 and 1"),
            (
                {"truncate": (0.9, 0.1), "truncate_scale": "quantile"},
                "quantile bounds need",
            ),
            ({"truncate": (0.1, 0.9), "truncate_scale": "nope"}, "truncate_scale"),
            ({"truncate": (0.9, 0.1)}, "truncate needs lower < upper"),
            ({"estimand": "ATT", "stabilize": True}, "defined for ATE weights only"),
            ({"exposure": "nope"}, "exposure must be"),
            ({"exposure": "continuous", "estimand": "ATT"}, "no treated or untreated"),
            (
                {"exposure": "continuous", "truncate": (0.1, 0.9)},
                "bounds a probability",
            ),
            ({"exposure": "continuous"}, "needs sigma="),
            ({"exposure": "continuous", "sigma": -1.0}, "needs sigma="),
        ],
    )
    def test_errors(self, kw, msg):
        args = {"ps": [0.3, 0.5, 0.7], "treat": [0, 1, 1], **kw}
        with pytest.raises(MethodIncompatibility, match=msg):
            sp.ps_weights(**args)


class TestEss:
    def test_overall_and_by_group(self):
        rng = np.random.default_rng(0)
        w = rng.uniform(0.1, 3, 30)
        g = np.array(["a", "b", "c"])[rng.integers(0, 3, 30)]
        assert sp.ess(w) == pytest.approx(w.sum() ** 2 / np.sum(w**2), rel=1e-14)
        by = sp.ess(w, by=g)
        assert list(by.index) == ["a", "b", "c"]
        for level in "abc":
            m = g == level
            assert by[level] == pytest.approx(w[m].sum() ** 2 / np.sum(w[m] ** 2))
        assert sp.ess(np.ones(17)) == pytest.approx(17.0)
        assert np.isnan(sp.ess(np.zeros(4)))

    def test_errors(self):
        with pytest.raises(MethodIncompatibility, match="missing values"):
            sp.ess([1.0, np.nan])
        with pytest.raises(MethodIncompatibility, match="same length"):
            sp.ess([1.0, 2.0], by=[1, 2, 3])
        with pytest.raises(MethodIncompatibility, match="weights is empty"):
            sp.ess([])


class TestEnergyDistance:
    def test_definition_with_and_without_weights(self):
        df = _data(n=80)
        df["b"] = (df.x1 > 0).astype(float)
        cols = ["x1", "b"]
        rng = np.random.default_rng(2)
        w = rng.uniform(0.2, 2, len(df))
        xv = df[cols].to_numpy()
        # continuous columns by their SD, 0/1 columns by sqrt(p (1 - p))
        p = df.b.mean()
        z = (xv - xv.mean(axis=0)) / np.array([df.x1.std(ddof=1), np.sqrt(p * (1 - p))])
        t = (df.d == 1).to_numpy()
        for wt in (np.ones(len(df)), w):
            a, b = wt[t] / wt[t].sum(), wt[~t] / wt[~t].sum()
            want = (
                2 * a @ cdist(z[t], z[~t]) @ b
                - a @ cdist(z[t], z[t]) @ a
                - b @ cdist(z[~t], z[~t]) @ b
            )
            got = sp.energy_distance(df, "d", cols, weights=wt)
            assert got == pytest.approx(want, rel=1e-10)
        assert sp.energy_distance(df, "d", cols) == pytest.approx(
            sp.energy_distance(df.assign(w=1.0), "d", cols, weights="w"), rel=1e-12
        )
        # the scale of the weights does not matter
        assert sp.energy_distance(df, "d", cols, weights=w) == pytest.approx(
            sp.energy_distance(df, "d", cols, weights=10 * w), rel=1e-10
        )

    def test_identical_groups_are_at_distance_zero(self):
        base = _data(n=40)[X]
        df = pd.concat([base.assign(d=1), base.assign(d=0)], ignore_index=True)
        # two copies of the same sample: every term cancels up to rounding
        assert sp.energy_distance(df, "d", X) == pytest.approx(0.0, abs=1e-12)

    def test_errors(self):
        df = _data(n=40)
        with pytest.raises(MethodIncompatibility, match="covariates is empty"):
            sp.energy_distance(df, "d", [])
        with pytest.raises(MethodIncompatibility, match="missing values"):
            sp.energy_distance(df.assign(x1=np.nan), "d", X)
        with pytest.raises(MethodIncompatibility, match="binary 0/1 with both"):
            sp.energy_distance(df.assign(d=1), "d", X)
        with pytest.raises(MethodIncompatibility, match="one entry per row"):
            sp.energy_distance(df, "d", X, weights=np.ones(3))
        with pytest.raises(MethodIncompatibility, match="finite and non-negative"):
            sp.energy_distance(df, "d", X, weights=-np.ones(40))
        with pytest.raises(MethodIncompatibility, match="positive total weight"):
            sp.energy_distance(df, "d", X, weights=df.d.to_numpy(dtype=float))


class TestImpliedWeights:
    def test_uniform_regression_weights_reproduce_the_coefficient(self):
        df = _data()
        w = sp.implied_weights(df, "d", X)
        ols = np.linalg.lstsq(
            np.column_stack([np.ones(len(df)), df.d, df[X]]), df.y, rcond=None
        )[0][1]
        t, c = df.d == 1, df.d == 0
        diff = np.average(df.y[t], weights=w[t]) - np.average(df.y[c], weights=w[c])
        assert diff == pytest.approx(ols, rel=1e-10)
        # scaled to the group sizes
        assert w[t].sum() == pytest.approx(t.sum(), rel=1e-10)
        assert w[c].sum() == pytest.approx(c.sum(), rel=1e-10)

    @pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATC", "ATU"])
    def test_arm_regression_weights_reproduce_regression_imputation(self, estimand):
        df = _data()
        w = sp.implied_weights(df, "d", X, interactions=True, estimand=estimand)
        t, c = (df.d == 1).to_numpy(), (df.d == 0).to_numpy()
        design = np.column_stack([np.ones(len(df)), df[X]])
        b1 = np.linalg.lstsq(design[t], df.y[t], rcond=None)[0]
        b0 = np.linalg.lstsq(design[c], df.y[c], rcond=None)[0]
        target = {"ATE": np.ones(len(df), bool), "ATT": t, "ATC": c, "ATU": c}[estimand]
        want = float(np.mean(design[target] @ (b1 - b0)))
        got = np.average(df.y[t], weights=w[t]) - np.average(df.y[c], weights=w[c])
        assert got == pytest.approx(want, rel=1e-10)

    def test_errors(self):
        df = _data(n=40)
        with pytest.raises(MethodIncompatibility, match="missing values"):
            sp.implied_weights(df.assign(x1=np.nan), "d", X)
        with pytest.raises(MethodIncompatibility, match="binary 0/1 with both"):
            sp.implied_weights(df.assign(d=0), "d", X)
        with pytest.raises(MethodIncompatibility, match="estimand must be"):
            sp.implied_weights(df, "d", X, estimand="ATO")
        with pytest.raises(MethodIncompatibility, match="single set of weights"):
            sp.implied_weights(df, "d", X, estimand="ATT")
        with pytest.raises(MethodIncompatibility, match="collinear within"):
            sp.implied_weights(df.assign(x3=df.x1), "d", X + ["x3"], interactions=True)


# ----------------------------------------------------------------------
# Optimal full matching
# ----------------------------------------------------------------------


def _best_edge_cover(cost: np.ndarray) -> float:
    """Smallest total cost over every set of edges that touches every node."""
    n1, n0 = cost.shape
    edges = [(i, j) for i in range(n1) for j in range(n0) if np.isfinite(cost[i, j])]
    best = np.inf
    for mask in range(1, 1 << len(edges)):
        rows, cols, total = 0, 0, 0.0
        for k, (i, j) in enumerate(edges):
            if mask >> k & 1:
                rows |= 1 << i
                cols |= 1 << j
                total += cost[i, j]
        if rows == (1 << n1) - 1 and cols == (1 << n0) - 1:
            best = min(best, total)
    return best


class TestFullMatchSets:
    @pytest.mark.parametrize("seed", range(6))
    def test_total_distance_is_the_minimum_over_all_edge_covers(self, seed):
        rng = np.random.default_rng(seed)
        n1, n0 = [(3, 4), (4, 3), (2, 5), (3, 3), (4, 2), (1, 4)][seed]
        cost = rng.uniform(0, 1, (n1, n0))
        set_t, set_c, total = full_match_sets(cost)
        # same sum over the same edges in the optimum
        assert total == pytest.approx(_best_edge_cover(cost), rel=1e-12)
        assert (set_t >= 0).all() and (set_c >= 0).all()
        # every set is a star: one treated, or one control
        for s in np.unique(set_t):
            assert (set_t == s).sum() == 1 or (set_c == s).sum() == 1
        # the total is the within-set distance to the star's centre
        check = 0.0
        for s in np.unique(set_t):
            ti, cj = np.flatnonzero(set_t == s), np.flatnonzero(set_c == s)
            check += cost[np.ix_(ti, cj)].sum()
        assert check == pytest.approx(total, rel=1e-12)

    def test_zero_distance_ties_still_give_stars(self):
        cost = np.zeros((3, 3))
        set_t, set_c, total = full_match_sets(cost)
        assert total == 0.0
        for s in np.unique(set_t):
            assert (set_t == s).sum() == 1 or (set_c == s).sum() == 1

    def test_caliper_leaves_units_without_a_partner_unmatched(self):
        cost = np.array([[0.1, 0.9, 5.0], [0.8, 0.2, 5.0], [7.0, 7.0, 6.0]])
        set_t, set_c, total = full_match_sets(cost, caliper=1.0)
        assert set_t[2] == -1 and set_c[2] == -1
        assert total == pytest.approx(0.3)
        none_t, none_c, zero = full_match_sets(cost, caliper=0.01)
        assert (none_t == -1).all() and (none_c == -1).all() and zero == 0.0
        # inf forbids a pairing without a caliper
        forbid = cost.copy()
        forbid[2, :] = np.inf
        assert full_match_sets(forbid)[0][2] == -1

    def test_errors(self):
        with pytest.raises(MethodIncompatibility, match="non-empty 2-D matrix"):
            full_match_sets(np.zeros(3))
        with pytest.raises(MethodIncompatibility, match="non-empty 2-D matrix"):
            full_match_sets(np.zeros((0, 3)))
        with pytest.raises(MethodIncompatibility, match="non-negative and not NaN"):
            full_match_sets(np.array([[1.0, np.nan]]))
        with pytest.raises(MethodIncompatibility, match="non-negative and not NaN"):
            full_match_sets(np.array([[1.0, -1.0]]))


class TestMatchedSetEffect:
    def _sets(self, seed=0, n=60, g=12):
        rng = np.random.default_rng(seed)
        sets = np.repeat(np.arange(g), n // g)
        d = np.zeros(n)
        # set k has one treated and several controls, or the reverse
        for k in range(g):
            idx = np.flatnonzero(sets == k)
            if k % 2:
                d[idx[1:]] = 1
            else:
                d[idx[0]] = 1
        y = sets * 0.3 + 1.5 * d + rng.normal(size=n)
        return y, d, sets

    @pytest.mark.parametrize("estimand", ["ATT", "ATC", "ATE"])
    def test_estimate_is_the_stratified_difference(self, estimand):
        y, d, sets = self._sets()
        num = den = 0.0
        for s in np.unique(sets):
            m = sets == s
            n1, n0 = d[m].sum(), (1 - d[m]).sum()
            size = {"ATT": n1, "ATC": n0, "ATE": n1 + n0}[estimand]
            num += size * (y[m][d[m] == 1].mean() - y[m][d[m] == 0].mean())
            den += size
        out = matched_set_effect(y, d, sets, estimand)
        assert out["estimate"] == pytest.approx(num / den, rel=1e-12)
        assert out["n_sets"] == 12 and out["n"] == 60

    def test_standard_error_is_the_cluster_robust_wls_one(self):
        sm = pytest.importorskip("statsmodels.api")
        y, d, sets = self._sets()
        out = matched_set_effect(y, d, sets, "ATT")
        ref = sm.WLS(y, sm.add_constant(d), weights=out["weights"]).fit(
            cov_type="cluster", cov_kwds={"groups": sets}
        )
        assert out["estimate"] == pytest.approx(ref.params[1], rel=1e-12)
        # statsmodels applies the same G/(G-1) (N-1)/(N-K) correction
        assert out["se"] == pytest.approx(ref.bse[1], rel=1e-10)

    def test_unmatched_rows_and_errors(self):
        y, d, sets = self._sets()
        dropped = sets.copy()
        dropped[sets == 11] = -1
        out = matched_set_effect(y, d, dropped)
        ref = matched_set_effect(y[sets != 11], d[sets != 11], sets[sets != 11])
        assert out["estimate"] == pytest.approx(ref["estimate"], rel=1e-12)
        assert out["kept"].sum() == 55
        with pytest.raises(MethodIncompatibility, match="ATT, ATC or ATE"):
            matched_set_effect(y, d, sets, "ATO")
        with pytest.raises(DataInsufficient, match="at least two sets"):
            matched_set_effect(y, d, np.zeros_like(sets))
        with pytest.raises(DataInsufficient, match="needs a treated and a"):
            matched_set_effect(y, np.zeros_like(d), sets)


class TestFullMatch:
    def test_given_score_matching_weights_and_balance(self):
        df = _data(n=120)
        df["ps"] = sp.propensity_score(df, "d", X)
        fit = sp.full_match(df, "y", "d", X, pscore="ps")
        t, c = df[df.d == 1], df[df.d == 0]
        cost = np.abs(t.ps.to_numpy()[:, None] - c.ps.to_numpy()[None, :])
        _, _, total = full_match_sets(cost)
        assert fit.total_distance == pytest.approx(total, rel=1e-12)
        assert fit.n_unmatched == 0
        assert fit.n_treated == len(t) and fit.n_control == len(c)
        ref = matched_set_effect(
            df.y.to_numpy(), df.d.to_numpy(), fit.subclass.to_numpy(dtype=int)
        )
        assert fit.estimate == pytest.approx(ref["estimate"], rel=1e-12)
        assert fit.att == fit.estimate
        assert fit.se == pytest.approx(ref["se"], rel=1e-12)
        z = stats.norm.ppf(0.975)
        assert fit.ci == pytest.approx(
            (fit.estimate - z * fit.se, fit.estimate + z * fit.se)
        )
        assert fit.pvalue == pytest.approx(
            2 * stats.norm.sf(abs(fit.estimate / fit.se)), rel=1e-10
        )
        # balance: weighted mean difference over the treated-group SD
        w = fit.weights.to_numpy()
        for v in X:
            x = df[v].to_numpy()
            sd1 = x[df.d == 1].std(ddof=1)
            after = (
                np.average(x[df.d == 1], weights=w[df.d == 1])
                - np.average(x[df.d == 0], weights=w[df.d == 0])
            ) / sd1
            assert fit.balance.loc[v, "smd_matched"] == pytest.approx(after, rel=1e-10)
            assert fit.balance.loc[v, "smd_unmatched"] == pytest.approx(
                (x[df.d == 1].mean() - x[df.d == 0].mean()) / sd1, rel=1e-12
            )
        wc = w[df.d == 0]
        assert fit.model_info["ess_control"] == pytest.approx(
            wc.sum() ** 2 / np.sum(wc**2), rel=1e-12
        )
        sizes = fit.subclass.value_counts()
        assert fit.model_info["set_sizes"]["max"] == int(sizes.max())
        assert fit.n_sets == len(sizes)

    def test_atc_is_the_att_of_the_relabelled_problem(self):
        df = _data(n=120)
        df["ps"] = sp.propensity_score(df, "d", X)
        atc = sp.full_match(df, "y", "d", X, pscore="ps", estimand="ATC")
        swapped = sp.full_match(df.assign(d=1 - df.d), "y", "d", X, pscore="ps")
        # the cost matrix is transposed, the optimum is the same sets
        assert swapped.total_distance == pytest.approx(atc.total_distance, rel=1e-12)
        assert swapped.estimate == pytest.approx(-atc.estimate, rel=1e-10)
        assert swapped.se == pytest.approx(atc.se, rel=1e-10)
        ate = sp.full_match(df, "y", "d", X, pscore="ps", estimand="ATE")
        ref = matched_set_effect(
            df.y.to_numpy(), df.d.to_numpy(), ate.subclass.to_numpy(dtype=int), "ATE"
        )
        assert ate.estimate == pytest.approx(ref["estimate"], rel=1e-12)

    @pytest.mark.parametrize(
        "distance", ["propensity", "logit", "mahalanobis", "euclidean"]
    )
    def test_distances(self, distance):
        sm = pytest.importorskip("statsmodels.api")
        df = _data(n=100)
        fit = sp.full_match(df, "y", "d", X, distance=distance)
        xv, d = df[X].to_numpy(), df.d.to_numpy()
        if distance in ("propensity", "logit"):
            res = sm.Logit(df.d, sm.add_constant(df[X])).fit(disp=0)
            score = np.asarray(
                res.predict() if distance == "propensity" else res.fittedvalues
            )
            cost = np.abs(score[d == 1][:, None] - score[d == 0][None, :])
            tol = 1e-6  # two logit fits
        elif distance == "mahalanobis":
            x1, x0 = xv[d == 1], xv[d == 0]
            cov = ((len(x1) - 1) * np.cov(x1.T) + (len(x0) - 1) * np.cov(x0.T)) / (
                len(xv) - 2
            )
            cost = cdist(x1, x0, metric="mahalanobis", VI=np.linalg.inv(cov))
            # the shared helper adds a 1e-8 ridge to the covariance
            tol = 1e-6
        else:
            cost = None
            tol = None
        if cost is not None:
            assert fit.total_distance == pytest.approx(
                full_match_sets(cost)[2], rel=tol
            )
        assert fit.n_unmatched == 0
        assert distance in fit.distance or "propensity" in fit.distance

    def test_probit_score_and_caliper_scales(self):
        df = _data(n=120)
        probit = sp.full_match(df, "y", "d", X, ps_model="probit")
        assert "probit" in probit.distance
        df["ps"] = sp.propensity_score(df, "d", X)
        sd = df.ps.std(ddof=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = sp.full_match(df, "y", "d", X, pscore="ps", caliper=0.05)
            b = sp.full_match(
                df, "y", "d", X, pscore="ps", caliper=0.05 * sd, caliper_scale="raw"
            )
        assert a.n_unmatched == b.n_unmatched
        assert a.estimate == pytest.approx(b.estimate, rel=1e-12)
        assert a.model_info["caliper"] == pytest.approx(0.05 * sd)

    def test_caliper_reports_the_units_it_leaves_out(self):
        df = _data(n=120)
        df["ps"] = sp.propensity_score(df, "d", X)
        with pytest.warns(RuntimeWarning, match="no partner within the caliper"):
            fit = sp.full_match(
                df, "y", "d", X, pscore="ps", caliper=0.002, caliper_scale="raw"
            )
        assert fit.n_unmatched > 0
        assert int(fit.subclass.isna().sum()) == fit.n_unmatched
        assert (fit.weights[fit.subclass.isna()] == 0).all()
        md = fit.matched_data(df)
        assert len(md) == len(df) - fit.n_unmatched
        assert {"subclass", "weights"} <= set(md.columns)
        with pytest.raises(DataInsufficient, match="leaves no unit with a partner"):
            sp.full_match(
                df, "y", "d", X, pscore="ps", caliper=1e-12, caliper_scale="raw"
            )

    def test_without_an_outcome_only_the_matching_is_returned(self):
        df = _data(n=100)
        fit = sp.full_match(df, treat="d", covariates=X)
        assert np.isnan(fit.estimate) and np.isnan(fit.se) and np.isnan(fit.pvalue)
        assert fit.n_sets > 1
        text = fit.summary()
        assert "Matched sets" in text and "SE =" not in text
        with_y = sp.full_match(df, "y", "d", X)
        assert "clustered on matched set" in with_y.summary()
        assert repr(with_y) == with_y.summary()
        # a single covariate may be named without a list
        assert sp.full_match(df, "y", "d", "x1").model_info["covariates"] == ["x1"]

    def test_missing_rows_are_dropped(self):
        df = _data(n=100)
        df.loc[[2, 7], "x2"] = np.nan
        fit = sp.full_match(df, "y", "d", X)
        ref = sp.full_match(df.dropna(), "y", "d", X)
        assert fit.estimate == pytest.approx(ref.estimate, rel=1e-12)
        assert len(fit.weights) == 98

    @pytest.mark.parametrize(
        "kw, exc, msg",
        [
            ({"treat": None}, MethodIncompatibility, "treat= and covariates="),
            ({"estimand": "ATO"}, MethodIncompatibility, "estimand must be"),
            ({"distance": "nope"}, MethodIncompatibility, "unknown distance"),
            ({"caliper_scale": "nope"}, MethodIncompatibility, "caliper_scale"),
            ({"covariates": ["zz"]}, MethodIncompatibility, "not in data"),
            ({"ps_model": "nope"}, MethodIncompatibility, "ps_model must be"),
            (
                {"distance": "mahalanobis", "caliper": 0.5},
                MethodIncompatibility,
                "needs a score distance",
            ),
        ],
    )
    def test_errors(self, kw, exc, msg):
        args = {"y": "y", "treat": "d", "covariates": X, **kw}
        with pytest.raises(exc, match=msg):
            sp.full_match(_data(n=60), **args)

    def test_treatment_errors(self):
        df = _data(n=60)
        with pytest.raises(MethodIncompatibility, match="coded 0/1"):
            sp.full_match(df.assign(d=np.arange(60) % 3), "y", "d", X)
        with pytest.raises(DataInsufficient, match="both treated and comparison"):
            sp.full_match(df.assign(d=1), "y", "d", X)


# ----------------------------------------------------------------------
# sp.pscore
# ----------------------------------------------------------------------


class TestPscore:
    def test_score_blocks_and_balancing_tests(self):
        sm = pytest.importorskip("statsmodels.api")
        df = _data(n=400)
        res = sp.pscore(df, "d", X, level=0.05)
        ref = sm.Logit(df.d, sm.add_constant(df[X])).fit(disp=0)
        np.testing.assert_allclose(res.pscore, ref.predict(), atol=1e-8)
        np.testing.assert_allclose(
            res.coefficients.loc[X + ["_cons"], "coef"],
            ref.params[X + ["const"]],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            res.coefficients.loc[X + ["_cons"], "se"], ref.bse[X + ["const"]], rtol=1e-5
        )
        assert res.loglik == pytest.approx(ref.llf, rel=1e-9)

        # each score lies in the interval of its block (bounds are stored
        # in single precision, as Stata stores them)
        info = res.blocks.set_index("block")
        ps, blk = res.pscore.to_numpy(), res.block.to_numpy()
        for b in info.index:
            inside = ps[blk == b]
            assert inside.min() >= info.loc[b, "lower"] - 1e-7
            assert inside.max() <= info.loc[b, "upper"] + 1e-7
            assert info.loc[b, "n_treated"] == int((df.d[blk == b] == 1).sum())
            assert info.loc[b, "n_control"] == int((df.d[blk == b] == 0).sum())
        assert res.n_blocks == int(np.nanmax(blk))

        # in every final block with both arms the mean score does not
        # differ at the chosen level, and the reported imbalances are
        # exactly the covariate t tests that reject
        rejected = []
        for b in info.index:
            m = blk == b
            t, c = m & (df.d == 1).to_numpy(), m & (df.d == 0).to_numpy()
            if not t.any() or not c.any():
                continue
            p_score = stats.ttest_ind(ps[t], ps[c]).pvalue
            assert not p_score < 0.05
            for v in X:
                p = stats.ttest_ind(df[v].to_numpy()[t], df[v].to_numpy()[c]).pvalue
                if p < 0.05:
                    rejected.append((v, b))
        got = list(zip(res.unbalanced["variable"], res.unbalanced["block"]))
        assert sorted(got) == sorted(rejected)
        assert res.balanced == (len(rejected) == 0)

    def test_common_support_is_the_range_of_the_treated_scores(self):
        df = _data(n=400)
        full = sp.pscore(df, "d", X)
        res = sp.pscore(df, "d", X, common_support=True, ps_model="probit")
        ps = res.pscore.to_numpy()
        lo, hi = ps[df.d == 1].min(), ps[df.d == 1].max()
        assert res.support_range == pytest.approx((lo, hi))
        inside = (ps >= lo) & (ps <= hi)
        assert 0 < (~inside).sum()
        assert res.support.to_numpy().tolist() == inside.tolist()
        assert res.block.isna().to_numpy().tolist() == (~inside).tolist()
        assert full.support.all() and full.block.notna().all()
        text = str(res.summary())
        assert "Region of common support" in text and "probit" in text
        assert "Region of common support" not in str(full.summary())

    def test_summary_lists_the_unbalanced_covariates(self):
        df = _data(n=600)
        # a lax level makes the within-block t tests reject (3 times here)
        res = sp.pscore(df, "d", X, level=0.2)
        assert not res.balanced and len(res.unbalanced) > 0
        row = res.unbalanced.iloc[0]
        assert f"{row.variable} is not balanced in block {row.block}" in str(
            res.summary()
        )
        ok = sp.pscore(df, "d", X, level=1e-12)
        assert ok.balanced
        assert "The balancing property is satisfied" in str(ok.summary())

    def test_assign_and_missing_rows(self):
        df = _data(n=200)
        df.loc[[3, 8], "x1"] = np.nan
        res = sp.pscore(df, "d", X)
        assert res.n_obs == 198
        assert res.pscore.index.equals(df.index)
        assert res.pscore.loc[[3, 8]].isna().all()
        assert not res.support.loc[[3, 8]].any()
        out = res.assign(df)
        assert {"pscore", "block", "comsup"} <= set(out.columns)
        slim = res.assign(df, pscore="e", block=None, support=None)
        assert "e" in slim.columns and "block" not in slim.columns
        assert "comsup" not in slim.columns

    def test_blocks_feed_stratification_matching(self):
        df = _data(n=400)
        res = sp.pscore(df, "d", X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = sp.match(
                res.assign(df),
                y="y",
                treat="d",
                covariates=X,
                method="stratify",
                strata="block",
            )
        num = den = 0.0
        for _, g in df.groupby(res.block):
            gt, gc = g[g.d == 1], g[g.d == 0]
            if len(gt) and len(gc):
                num += len(gt) * (gt.y.mean() - gc.y.mean())
                den += len(gt)
        assert got.estimate == pytest.approx(num / den, rel=1e-12)

    @pytest.mark.parametrize(
        "kw, exc, msg",
        [
            ({"data": [1, 2]}, MethodIncompatibility, "must be a pandas DataFrame"),
            ({"covariates": ["zz"]}, MethodIncompatibility, "columns not found"),
            ({"covariates": []}, MethodIncompatibility, "at least one covariate"),
            ({"ps_model": "nope"}, MethodIncompatibility, "ps_model must be"),
            ({"level": 1.0}, MethodIncompatibility, "level must lie strictly"),
            ({"n_blocks": 0}, MethodIncompatibility, "n_blocks must be a positive"),
        ],
    )
    def test_errors(self, kw, exc, msg):
        args = {"data": _data(n=60), "treat": "d", "covariates": X, **kw}
        with pytest.raises(exc, match=msg):
            sp.pscore(**args)

    def test_treatment_errors(self):
        df = _data(n=60)
        with pytest.raises(MethodIncompatibility, match=r"must be binary \(0/1\)"):
            sp.pscore(df.assign(d=np.arange(60) % 3), "d", X)
        with pytest.raises(DataInsufficient, match="both treated and control"):
            sp.pscore(df.assign(d=0), "d", X)


# ----------------------------------------------------------------------
# sp.cbps
# ----------------------------------------------------------------------


class TestCbps:
    def _ps(self, df, beta):
        design = np.column_stack([np.ones(len(df)), df[X]])
        return 1 / (1 + np.exp(-design @ beta))

    @pytest.mark.parametrize("estimand", ["ATE", "ATT"])
    def test_exact_variant_balances_the_covariate_means(self, estimand):
        df = _data(n=250)
        res = sp.cbps(
            df, "y", "d", X, estimand=estimand, variant="exact", n_bootstrap=2, seed=0
        )
        ps = self._ps(df, res.model_info["beta"])
        t, y = df.d.to_numpy(dtype=float), df.y.to_numpy()
        if estimand == "ATE":
            w1, w0 = t / ps, (1 - t) / (1 - ps)
        else:
            w1, w0 = t, (1 - t) * ps / (1 - ps)
        w1, w0 = w1 / w1.sum(), w0 / w0.sum()
        # the just-identified moments are solved to the optimiser's
        # tolerance, so the weighted covariate means agree across arms
        for v in X:
            assert w1 @ df[v] == pytest.approx(w0 @ df[v], abs=1e-5)
            assert res.model_info["std_mean_diff_after"][v] == pytest.approx(
                0.0, abs=1e-4
            )
        assert res.estimate == pytest.approx(float(w1 @ y - w0 @ y), abs=1e-8)
        assert res.model_info["converged"] is True
        assert res.estimand == estimand

    def test_over_identified_estimate_follows_from_its_coefficients(self):
        df = _data(n=250)
        res = sp.cbps(df, "y", "d", X, n_bootstrap=2, seed=0)
        ps = self._ps(df, res.model_info["beta"])
        t, y = df.d.to_numpy(dtype=float), df.y.to_numpy()
        w1, w0 = t / ps, (1 - t) / (1 - ps)
        assert res.estimate == pytest.approx(
            float(w1 @ y / w1.sum() - w0 @ y / w0.sum()), abs=1e-8
        )
        assert res.model_info["pscore_min"] == pytest.approx(ps.min(), abs=1e-8)
        assert res.model_info["n_treated"] == int(t.sum())

    def test_over_identified_att(self):
        df = _data(n=250)
        res = sp.cbps(df, "y", "d", X, estimand="ATT", n_bootstrap=2, seed=0)
        ps = self._ps(df, res.model_info["beta"])
        t, y = df.d.to_numpy(dtype=float), df.y.to_numpy()
        w0 = (1 - t) * ps / (1 - ps)
        assert res.estimate == pytest.approx(
            float(y[t == 1].mean() - w0 @ y / w0.sum()), abs=1e-8
        )
        assert res.method == "CBPS (over, ATT)"

    def test_trim_clips_the_scores_before_weighting(self):
        df = _data(n=250)
        res = sp.cbps(df, "y", "d", X, n_bootstrap=2, seed=0, trim=0.3)
        ps = np.clip(self._ps(df, res.model_info["beta"]), 0.3, 0.7)
        t, y = df.d.to_numpy(dtype=float), df.y.to_numpy()
        w1, w0 = t / ps, (1 - t) / (1 - ps)
        assert res.estimate == pytest.approx(
            float(w1 @ y / w1.sum() - w0 @ y / w0.sum()), abs=1e-8
        )
        assert res.model_info["pscore_min"] == pytest.approx(0.3)
        assert res.model_info["pscore_max"] == pytest.approx(0.7)

    def test_bootstrap_se_is_the_sd_of_refits_on_resamples(self):
        df = _data(n=200)
        reps, seed = 4, 11
        res = sp.cbps(df, "y", "d", X, variant="exact", n_bootstrap=reps, seed=seed)
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(reps):
            idx = rng.integers(0, len(df), size=len(df))
            rep = sp.cbps(
                df.iloc[idx].reset_index(drop=True),
                "y",
                "d",
                X,
                variant="exact",
                n_bootstrap=2,
                seed=0,
            )
            draws.append(rep.estimate)
        assert res.se == pytest.approx(np.std(draws, ddof=1), rel=1e-8)
        assert res.model_info["n_bootstrap_success"] == reps
        z = stats.norm.ppf(0.975)
        assert res.ci == pytest.approx(
            (res.estimate - z * res.se, res.estimate + z * res.se)
        )
        assert res.pvalue == pytest.approx(
            2 * stats.norm.sf(abs(res.estimate) / res.se), rel=1e-10
        )

    def test_without_replications_there_is_no_standard_error(self):
        df = _data(n=200)
        for reps in (0, 1):
            res = sp.cbps(df, "y", "d", X, variant="exact", n_bootstrap=reps, seed=0)
            assert np.isnan(res.se) and np.isnan(res.pvalue)
            assert np.isnan(res.ci[0]) and np.isnan(res.ci[1])

    def test_a_caller_supplied_constant_replaces_the_intercept(self):
        df = _data(n=200).assign(one=1.0)
        a = sp.cbps(df, "y", "d", X, variant="exact", n_bootstrap=2, seed=0)
        b = sp.cbps(
            df,
            "y",
            "d",
            ["one"] + X,
            variant="exact",
            n_bootstrap=2,
            seed=0,
            add_intercept=False,
        )
        assert b.estimate == pytest.approx(a.estimate, abs=1e-8)
        assert sorted(b.model_info["std_mean_diff_after"]) == ["one", "x1", "x2"]

    def test_errors(self):
        df = _data(n=100)
        with pytest.raises(ValueError, match="estimand must be 'ATE' or 'ATT'"):
            sp.cbps(df, "y", "d", X, estimand="ATC")
        with pytest.raises(ValueError, match="variant must be 'exact' or 'over'"):
            sp.cbps(df, "y", "d", X, variant="nope")
        with pytest.raises(ValueError, match="exactly one constant"):
            sp.cbps(df, "y", "d", X, add_intercept=False, n_bootstrap=2)
        with pytest.raises(ValueError, match="exactly one constant"):
            sp.cbps(df.assign(one=1.0), "y", "d", X + ["one"], n_bootstrap=2)

    def test_a_redundant_covariate_is_refused_or_changes_nothing(self):
        # The model with x3 = x1 is the model without it. On this sample
        # the just-identified estimate moves from 1.884 to 1.993.
        df = _data(seed=2)
        base = sp.cbps(df, "y", "d", X, variant="exact", n_bootstrap=2, seed=0)
        try:
            extra = sp.cbps(
                df.assign(x3=df.x1),
                "y",
                "d",
                X + ["x3"],
                variant="exact",
                n_bootstrap=2,
                seed=0,
            )
        except ValueError as exc:
            assert "not full rank" in str(exc)
            return
        assert extra.estimate == pytest.approx(base.estimate, abs=1e-6)


# ----------------------------------------------------------------------
# Two-criteria matching
# ----------------------------------------------------------------------


def _two_criteria_optimum(pair, balance, use, ratio, skip=None):
    """Brute force over every way of giving each treated unit its controls."""
    n_t, n_c = pair.shape
    best = np.inf
    options = [list(itertools.combinations(range(n_c), ratio))] * n_t
    if skip is not None:
        options = [opts + [None] for opts in options]
    for choice in itertools.product(*options):
        chosen = [j for c in choice if c is not None for j in c]
        if len(set(chosen)) != len(chosen):
            continue
        kept = [i for i, c in enumerate(choice) if c is not None]
        total = sum(pair[i, j] for i in kept for j in choice[i])
        total += sum(use[j] for j in chosen)
        if skip is not None:
            total += skip * (n_t - len(kept))
        if chosen:
            # balance: the cheapest re-assignment of the chosen controls,
            # `ratio` to each kept treated unit, ignoring the pairing
            slots = [i for i in kept for _ in range(ratio)]
            total += min(
                sum(balance[i, j] for i, j in zip(slots, perm))
                for perm in itertools.permutations(chosen)
            )
        best = min(best, total)
    return best


class TestTwoCriteria:
    def _frame(self, n_t=3, n_c=5):
        return pd.DataFrame({"z": [1] * n_t + [0] * n_c, "v": np.arange(n_t + n_c)})

    @pytest.mark.parametrize("seed", range(4))
    def test_total_cost_is_the_minimum_over_all_matches(self, seed):
        rng = np.random.default_rng(seed)
        pair = rng.integers(0, 20, (3, 5)).astype(float)
        balance = rng.integers(0, 20, (3, 5)).astype(float)
        use = rng.integers(0, 5, 5).astype(float)
        res = sp.two_criteria_match(
            self._frame(), "z", pair=pair, balance=balance, control_cost=use
        )
        want = _two_criteria_optimum(pair, balance, use, 1)
        assert res.total_cost == pytest.approx(want, abs=1e-9)
        assert res.pair_cost + res.balance_cost + res.control_cost == pytest.approx(
            res.total_cost
        )
        assert res.n_sets == 3 and res.n_unmatched == 0
        assert res.matched.groupby("mset").size().tolist() == [2, 2, 2]
        assert res.pairs["control"].is_unique
        # the first row of every set is its treated unit
        assert (res.matched.groupby("mset")["z"].first() == 1).all()

    def test_two_controls_per_treated_unit(self):
        rng = np.random.default_rng(5)
        pair = rng.integers(0, 20, (2, 5)).astype(float)
        balance = rng.integers(0, 20, (2, 5)).astype(float)
        res = sp.two_criteria_match(
            self._frame(2, 5), "z", pair=pair, balance=balance, ratio=2
        )
        want = _two_criteria_optimum(pair, balance, np.zeros(5), 2)
        assert res.total_cost == pytest.approx(want, abs=1e-9)
        assert res.matched.groupby("mset").size().tolist() == [3, 3]
        assert "ratio=2" in repr(res)

    def test_subset_cost_drops_treated_units_that_match_badly(self):
        pair = np.array(
            [[1.0, 2.0, 3.0, 9.0], [2.0, 1.0, 3.0, 9.0], [90.0, 80.0, 70.0, 95.0]]
        )
        frame = self._frame(3, 4)
        res = sp.two_criteria_match(frame, "z", pair=pair, subset_cost=10.0)
        assert res.total_cost == pytest.approx(
            _two_criteria_optimum(pair, np.zeros((3, 4)), np.zeros(4), 1, skip=10.0)
        )
        assert res.n_unmatched == 1 and res.n_sets == 2
        assert res.diagnostics["skip_cost"] == 10.0
        assert 2 not in res.matched.index
        assert "1 treated left out" in res.summary()
        every = sp.two_criteria_match(frame, "z", pair=pair)
        assert every.n_unmatched == 0 and every.pair_cost == pytest.approx(72.0)

    def test_rank_mahalanobis_for_one_untied_covariate(self):
        # one covariate without ties: the rank variance is n (n + 1) / 12,
        # so the distance is the squared rank difference over that
        rng = np.random.default_rng(0)
        x = rng.normal(size=9)
        treated = np.array([True, False, True, False, False, True, False, False, False])
        ranks = stats.rankdata(x)
        want = (ranks[treated][:, None] - ranks[~treated][None, :]) ** 2 / (9 * 10 / 12)
        np.testing.assert_allclose(rank_mahalanobis(x, treated), want, rtol=1e-12)
        with pytest.raises(MethodIncompatibility, match="is constant"):
            rank_mahalanobis(np.column_stack([x, np.ones(9)]), treated)

    def test_cost_terms(self):
        df = pd.DataFrame(
            {
                "z": [1, 1, 0, 0, 0, 0],
                "sex": ["f", "m", "f", "m", "m", "f"],
                "k": [1, 4, 1, 2, 4, 6],
                "v": [0.0, 1.0, 0.1, 0.5, 2.0, 3.5],
            }
        )
        t, c = df[df.z == 1], df[df.z == 0]
        near = (t.sex.to_numpy()[:, None] != c.sex.to_numpy()[None, :]) * 7.0
        integer = np.abs(t.k.to_numpy()[:, None] - c.k.to_numpy()[None, :]) * 2.0
        diff = t.v.to_numpy()[:, None] - c.v.to_numpy()[None, :]
        cal = ((np.abs(diff) > 0.4).astype(float) + (np.abs(diff) > 0.8)) * 5.0
        one_step = (np.abs(diff) > 0.4) * 5.0
        asym = ((diff > 0.2).astype(float) + (diff < -1.0)) * 5.0
        cuts = np.quantile(df.v, [0.25, 0.75])
        level = np.searchsorted(cuts, df.v.to_numpy(), side="left")
        quant = (
            np.abs(level[df.z == 1][:, None] - level[df.z == 0][None, :]) * 3.0
        ).astype(float)
        sd_half = 0.2 * df.v.std(ddof=1)
        default = (
            (np.abs(diff) > sd_half).astype(float) + (np.abs(diff) > 2 * sd_half)
        ) * 1000.0

        def cost(term):
            res = sp.two_criteria_match(df, "z", pair=term)
            got = np.full((2, 4), np.nan)
            for row in res.pairs.itertuples():
                got[row.treated, row.control - 2] = row.pair_cost
            return got

        for term, want in [
            ({"type": "near_exact", "on": "sex", "penalty": 7}, near),
            ({"type": "integer", "on": "k", "penalty": 2}, integer),
            ({"type": "caliper", "on": "v", "width": 0.4, "penalty": 5}, cal),
            (
                {
                    "type": "caliper",
                    "on": "v",
                    "width": 0.4,
                    "penalty": 5,
                    "two_step": False,
                },
                one_step,
            ),
            (
                {
                    "type": "caliper",
                    "on": "v",
                    "width": (-1.0, 0.2),
                    "penalty": 5,
                    "two_step": False,
                },
                asym,
            ),
            ({"type": "caliper", "on": "v"}, default),
            (
                {"type": "quantile", "on": "v", "probs": [0.25, 0.75], "penalty": 3},
                quant,
            ),
        ]:
            got = cost(term)
            used = np.isfinite(got)
            assert used.sum() == 2
            np.testing.assert_allclose(got[used], want[used], rtol=1e-12)
            # and the chosen pairs are a cheapest assignment under that cost
            assert got[used].sum() == pytest.approx(
                _two_criteria_optimum(want, np.zeros((2, 4)), np.zeros(4), 1)
            )

    def test_fitted_score_and_balance_table(self):
        sm = pytest.importorskip("statsmodels.api")
        df = _data(n=120)
        df["z"] = 1 - df.d  # 52 treated, 68 controls
        df = df.drop(columns="d")
        res = sp.two_criteria_match(
            df,
            "z",
            ps=X,
            pair=[{"type": "mahalanobis", "on": X}],
            balance={"type": "caliper", "on": "pscore", "penalty": 50},
        )
        ref = sm.Logit(df.z, sm.add_constant(df[X])).fit(disp=0).predict()
        np.testing.assert_allclose(
            res.matched["pscore"], ref[res.matched.index], atol=1e-8
        )
        m = res.matched
        for v in X:
            t_all, c_all = df.loc[df.z == 1, v], df.loc[df.z == 0, v]
            pooled = np.sqrt((t_all.var() + c_all.var()) / 2)
            row = res.balance.loc[v]
            assert row["smd_before"] == pytest.approx(
                (t_all.mean() - c_all.mean()) / pooled, rel=1e-12
            )
            assert row["smd_after"] == pytest.approx(
                (m.loc[m.z == 1, v].mean() - m.loc[m.z == 0, v].mean()) / pooled,
                rel=1e-12,
            )
            assert row["all_control"] == pytest.approx(c_all.mean())
        assert "pscore" in res.balance.index
        assert "Pairing cost" in res.summary()
        # a score given as a column is used as it is
        given = sp.two_criteria_match(
            df.assign(e=ref),
            "z",
            ps="e",
            pair={"type": "caliper", "on": "pscore", "width": 0.05},
        )
        np.testing.assert_allclose(given.matched["pscore"], ref[given.matched.index])
        # a control cost given by column name
        priced = sp.two_criteria_match(
            df.assign(price=1.0),
            "z",
            pair=[{"type": "mahalanobis", "on": X}],
            control_cost="price",
        )
        assert priced.control_cost == pytest.approx(float((df.z == 1).sum()))

    def test_rows_with_a_missing_treatment_are_ignored(self):
        df = _data(n=60).drop(columns="d")
        df["z"] = (df.index % 3 == 0).astype(float)
        df.loc[[0, 1], "z"] = np.nan
        res = sp.two_criteria_match(df, "z", pair=[{"type": "mahalanobis", "on": X}])
        ref = sp.two_criteria_match(
            df.dropna(subset=["z"]), "z", pair=[{"type": "mahalanobis", "on": X}]
        )
        assert res.total_cost == pytest.approx(ref.total_cost)
        assert not {0, 1} & set(res.matched.index)

    @pytest.mark.parametrize(
        "kw, exc, msg",
        [
            ({}, MethodIncompatibility, "at least one of pair= and balance="),
            ({"pair": np.zeros((2, 2))}, MethodIncompatibility, "must have shape"),
            ({"pair": -np.ones((3, 5))}, MethodIncompatibility, "finite and non-neg"),
            ({"pair": [{"on": "v"}]}, MethodIncompatibility, "dict with a 'type' key"),
            ({"pair": [{"type": "nope", "on": "v"}]}, MethodIncompatibility, "unknown"),
            (
                {"pair": [{"type": "integer", "on": "v", "width": 1}]},
                MethodIncompatibility,
                "takes the keys",
            ),
            ({"pair": [{"type": "integer"}]}, MethodIncompatibility, "takes the keys"),
            (
                {"pair": [{"type": "integer", "on": "v", "penalty": 0}]},
                MethodIncompatibility,
                "positive penalty",
            ),
            (
                {"pair": [{"type": "integer", "on": 3}]},
                MethodIncompatibility,
                "needs a column name",
            ),
            (
                {"pair": [{"type": "integer", "on": "s"}]},
                MethodIncompatibility,
                "needs a numeric column",
            ),
            (
                {"pair": [{"type": "integer", "on": "m"}]},
                MethodIncompatibility,
                "has missing values",
            ),
            (
                {"pair": [{"type": "quantile", "on": "v", "probs": [0.0, 0.5]}]},
                MethodIncompatibility,
                "probs strictly between",
            ),
            (
                {"pair": [{"type": "caliper", "on": "v", "width": (1, 2)}]},
                MethodIncompatibility,
                "low <= 0 <= high",
            ),
            ({"pair": np.zeros((3, 5)), "ratio": 0}, MethodIncompatibility, "ratio"),
            ({"pair": np.zeros((3, 5)), "ratio": 1.5}, MethodIncompatibility, "ratio"),
            (
                {"pair": np.zeros((3, 5)), "ratio": 2},
                DataInsufficient,
                "cannot give each",
            ),
            (
                {"pair": np.zeros((3, 5)), "ratio": 2, "subset_cost": 1.0},
                MethodIncompatibility,
                "subset_cost needs ratio=1",
            ),
            (
                {"pair": np.zeros((3, 5)), "subset_cost": -1.0},
                MethodIncompatibility,
                "subset_cost must be non-negative",
            ),
            (
                {"pair": np.zeros((3, 5)), "control_cost": [1.0, 2.0]},
                MethodIncompatibility,
                "one finite non-negative value per control",
            ),
            (
                {"pair": np.zeros((3, 5)), "control_cost": -np.ones(5)},
                MethodIncompatibility,
                "one finite non-negative value per control",
            ),
        ],
    )
    def test_errors(self, kw, exc, msg):
        df = self._frame().assign(s="a", m=[1.0, np.nan] + [0.0] * 6)
        with pytest.raises(exc, match=msg):
            sp.two_criteria_match(df, "z", **kw)

    def test_treatment_errors(self):
        df = self._frame()
        with pytest.raises(MethodIncompatibility, match="coded 0/1"):
            sp.two_criteria_match(df.assign(z=2), "z", pair=np.zeros((3, 5)))
        with pytest.raises(DataInsufficient, match="both treated units and controls"):
            sp.two_criteria_match(df.assign(z=1), "z", pair=np.zeros((8, 0)))


class TestTightenBlocks:
    def _blocks(self, n_blocks=6, per=4, seed=0):
        rng = np.random.default_rng(seed)
        return pd.DataFrame(
            {
                "block": np.repeat(np.arange(n_blocks), per),
                "z": np.tile([1] + [0] * (per - 1), n_blocks),
                "bmi": rng.normal(27, 4, n_blocks * per),
                "smoker": rng.integers(0, 2, n_blocks * per),
            }
        )

    def test_keeps_the_closest_control_of_each_block(self):
        df = self._blocks()
        res = sp.tighten_blocks(df, "z", "block", covariates=["bmi"])
        treated = (df.z == 1).to_numpy()
        dist = rank_mahalanobis(df.bmi.to_numpy(), treated)
        c_rows = np.flatnonzero(~treated)
        want = []
        for k, t_row in enumerate(np.flatnonzero(treated)):
            own = np.flatnonzero(df.block.to_numpy()[c_rows] == df.block[t_row])
            want.append(int(c_rows[own[dist[k, own].argmin()]]))
        assert sorted(res.pairs["control"]) == sorted(want)
        assert res.matched.groupby("mset")["block"].nunique().eq(1).all()
        assert res.matched.groupby("mset").size().tolist() == [2] * 6
        assert res.diagnostics["block_penalty"] > dist.max()

    def test_fine_balance_takes_priority_over_distance(self):
        df = self._blocks(n_blocks=8)
        res = sp.tighten_blocks(
            df, "z", "block", covariates=["bmi"], fine_balance=["smoker"]
        )
        m = res.matched
        gap = abs(
            int(m.loc[m.z == 1, "smoker"].sum()) - int(m.loc[m.z == 0, "smoker"].sum())
        )
        # the smallest gap any choice of one control per block can reach
        n_t1 = int(df.loc[df.z == 1, "smoker"].sum())
        can0 = df[df.z == 0].groupby("block")["smoker"].min().sum()
        can1 = df[df.z == 0].groupby("block")["smoker"].max().sum()
        best = 0 if can0 <= n_t1 <= can1 else min(abs(n_t1 - can0), abs(n_t1 - can1))
        assert gap == best
        only_fine = sp.tighten_blocks(
            df, "z", "block", fine_balance=["smoker"], ratio=2
        )
        assert only_fine.matched.groupby("mset").size().tolist() == [3] * 8

    def test_errors(self):
        df = self._blocks()
        with pytest.raises(MethodIncompatibility, match="covariates=, fine_balance="):
            sp.tighten_blocks(df, "z", "block")
        with pytest.raises(MethodIncompatibility, match="ratio must be a positive"):
            sp.tighten_blocks(df, "z", "block", covariates=["bmi"], ratio=0)
        with pytest.raises(MethodIncompatibility, match="penalty_scale must be"):
            sp.tighten_blocks(df, "z", "block", covariates=["bmi"], penalty_scale=0)
        with pytest.raises(MethodIncompatibility, match="subset_cost needs ratio=1"):
            sp.tighten_blocks(
                df, "z", "block", covariates=["bmi"], ratio=2, subset_cost=1.0
            )
        with pytest.raises(DataInsufficient, match="fewer than 4 controls"):
            sp.tighten_blocks(df, "z", "block", covariates=["bmi"], ratio=4)
        two_treated = df.copy()
        two_treated.loc[1, "z"] = 1
        with pytest.raises(MethodIncompatibility, match="exactly one treated"):
            sp.tighten_blocks(two_treated, "z", "block", covariates=["bmi"])
