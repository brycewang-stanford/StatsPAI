"""sp.match: every option against a computation written out in the test.

Written while scanning the branches of ``matching/match.py`` that the suite
did not reach. The passing tests pin behaviour against brute-force matching
on small samples, weighted means written with numpy, or an identity. The
tests under ``TestDefects`` state the correct behaviour where the code used
to return something else (they were strict xfails until it was fixed); the
classes after it check what replaced the defects against R ``Matching`` and
brute force.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.spatial.distance import cdist

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

X = ["x1", "x2"]


def _data(seed: int = 0, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2))))
    y = 1 + 2 * d + x1 + 0.5 * x2 + d * x1 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


def _small(seed: int = 1, n_t: int = 7, n_c: int = 12) -> pd.DataFrame:
    """A sample small enough to enumerate, with a score given as a column."""
    rng = np.random.default_rng(seed)
    n = n_t + n_c
    d = np.r_[np.ones(n_t, dtype=int), np.zeros(n_c, dtype=int)]
    ps = np.round(rng.uniform(0.1, 0.9, n), 3) + np.arange(n) * 1e-5
    x1 = rng.normal(size=n)
    y = 2.0 * d + 3.0 * ps + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "ps": ps, "x1": x1, "x2": rng.normal(size=n)})


def _fit(data: pd.DataFrame, covariates=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.match(data, y="y", treat="d", covariates=covariates or X, **kw)


def _nn_att(yt, yc, dist, k=1, caliper=None):
    """Brute-force k-NN ATT with every tie at the k-th distance kept."""
    effects = []
    for i in range(dist.shape[0]):
        row = dist[i].copy()
        if caliper is not None:
            row[row > caliper] = np.inf
        ok = np.isfinite(row)
        if not ok.any():
            continue
        kth = np.sort(row[ok])[min(k, ok.sum()) - 1]
        effects.append(yt[i] - yc[ok & (row <= kth)].mean())
    return float(np.mean(effects)), np.asarray(effects)


class TestNearestNeighbourAgainstBruteForce:
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_att_on_a_given_score(self, k):
        df = _small()
        t, c = df[df.d == 1], df[df.d == 0]
        dist = np.abs(t.ps.to_numpy()[:, None] - c.ps.to_numpy()[None, :])
        want, _ = _nn_att(t.y.to_numpy(), c.y.to_numpy(), dist, k=k)
        got = _fit(df, covariates=["x1"], pscore="ps", n_matches=k)
        # same arithmetic on the same pairs: only summation order differs
        assert got.estimate == pytest.approx(want, rel=1e-12)
        assert got.model_info["pscore_source"] == "given"

    def test_simple_pair_se_and_psmatch2_se(self):
        df = _small()
        t, c = df[df.d == 1], df[df.d == 0]
        dist = np.abs(t.ps.to_numpy()[:, None] - c.ps.to_numpy()[None, :])
        _, eff = _nn_att(t.y.to_numpy(), c.y.to_numpy(), dist)
        ai = _fit(df, covariates=["x1"], pscore="ps", se_method="ai")
        assert ai.se == pytest.approx(eff.std(ddof=1) / np.sqrt(len(eff)), rel=1e-12)

        # psmatch2: sqrt(var1/N1 + var0 * sum(w^2) / N1^2), w the number of
        # times a control is used, var0 over the controls that are used.
        use = np.bincount(dist.argmin(axis=1), minlength=len(c)).astype(float)
        n1 = len(t)
        want = np.sqrt(
            t.y.var(ddof=1) / n1
            + c.y.to_numpy()[use > 0].var(ddof=1) * np.sum(use**2) / n1**2
        )
        ps2 = _fit(df, covariates=["x1"], pscore="ps", se_method="psmatch2")
        assert ps2.se == pytest.approx(want, rel=1e-12)
        assert ps2.model_info["se_method"] == "psmatch2"

    def test_population_variance_formula(self):
        # Abadie-Imbens (2006) population ATT variance as the docstring of
        # _ai_population_se writes it, for one match per treated unit.
        df = _small()
        t, c = df[df.d == 1], df[df.d == 0]
        dist = np.abs(t.ps.to_numpy()[:, None] - c.ps.to_numpy()[None, :])
        j = dist.argmin(axis=1)
        tau_i = t.y.to_numpy() - c.y.to_numpy()[j]
        tau = tau_i.mean()
        k_j = np.bincount(j, minlength=len(c)).astype(float)
        sigma2 = 0.5 * np.mean((tau_i - tau) ** 2)
        var = (sigma2 * np.sum(k_j**2 - k_j) + np.sum((tau_i - tau) ** 2)) / len(t) ** 2
        got = _fit(df, covariates=["x1"], pscore="ps", se_method="abadie_imbens_pop")
        assert got.se == pytest.approx(np.sqrt(var), rel=1e-12)

    def test_caliper_drops_treated_units_and_counts_them(self):
        df = _small()
        t, c = df[df.d == 1], df[df.d == 0]
        dist = np.abs(t.ps.to_numpy()[:, None] - c.ps.to_numpy()[None, :])
        caliper = float(np.sort(dist.min(axis=1))[3])  # keeps 4 of the 7
        want, eff = _nn_att(t.y.to_numpy(), c.y.to_numpy(), dist, caliper=caliper)
        assert len(eff) == 4
        with pytest.warns(UserWarning, match="3 of 7 on-support treated"):
            got = sp.match(
                df, y="y", treat="d", covariates=["x1"], pscore="ps", caliper=caliper
            )
        assert got.estimate == pytest.approx(want, rel=1e-12)
        assert got.model_info["n_treated_unmatched"] == 3
        assert got.model_info["n_matched_treated"] == 4
        # unmatched treated carry no weight in the matched frame
        md = got.matched_data
        assert int(md.loc[md.d == 1, "_weight"].notna().sum()) == 4

    def test_caliper_in_sd_units_is_the_raw_caliper_times_sd_of_the_score(self):
        df = _small()
        sd = float(df.ps.std(ddof=1))
        a = _fit(df, covariates=["x1"], pscore="ps", caliper=0.3, caliper_scale="sd")
        b = _fit(df, covariates=["x1"], pscore="ps", caliper=0.3 * sd)
        assert a.estimate == pytest.approx(b.estimate, rel=1e-12)
        assert (
            a.model_info["n_treated_unmatched"] == b.model_info["n_treated_unmatched"]
        )

    def test_common_support_minmax_drops_treated_outside_the_control_range(self):
        df = _small()
        df.loc[0, "ps"] = 0.99  # one treated unit above every control
        t, c = df[df.d == 1], df[df.d == 0]
        on = (t.ps >= c.ps.min()) & (t.ps <= c.ps.max())
        assert not on.iloc[0] and on.any()
        dist = np.abs(t.ps.to_numpy()[on][:, None] - c.ps.to_numpy()[None, :])
        want, _ = _nn_att(t.y.to_numpy()[on], c.y.to_numpy(), dist)
        got = _fit(df, covariates=["x1"], pscore="ps", common_support="minmax")
        assert got.estimate == pytest.approx(want, rel=1e-12)
        assert got.model_info["n_treated_on_support"] == int(on.sum())
        assert got.matched_data.loc[0, "_support"] == 0

    @pytest.mark.parametrize("metric", ["euclidean", "mahalanobis", "total"])
    def test_covariate_distances(self, metric):
        df = _data(n=120)
        xv, d, y = df[X].to_numpy(), df.d.to_numpy(), df.y.to_numpy()
        if metric == "euclidean":
            z = xv / xv.std(axis=0, ddof=1)
            dist = cdist(z[d == 1], z[d == 0])
            kw = {"distance": "euclidean"}
        else:
            if metric == "total":
                cov = np.cov(xv.T)
                kw = {"distance": "mahalanobis", "mahalanobis_cov": "total"}
            else:
                x1, x0 = xv[d == 1], xv[d == 0]
                cov = ((len(x1) - 1) * np.cov(x1.T) + (len(x0) - 1) * np.cov(x0.T)) / (
                    len(xv) - 2
                )
                kw = {"distance": "mahalanobis"}
            dist = cdist(
                xv[d == 1], xv[d == 0], metric="mahalanobis", VI=np.linalg.inv(cov)
            )
        want, _ = _nn_att(y[d == 1], y[d == 0], dist)
        assert _fit(df, **kw).estimate == pytest.approx(want, rel=1e-12)

    def test_legacy_method_names(self):
        df = _data(n=120)
        assert _fit(df, method="psm").estimate == _fit(df).estimate
        assert (
            _fit(df, method="mahalanobis").estimate
            == _fit(df, distance="mahalanobis").estimate
        )


class TestTies:
    def _tied(self) -> pd.DataFrame:
        # two treated units, each with two controls at exactly its score
        return pd.DataFrame(
            {
                "y": [10.0, 20.0, 1.0, 3.0, 5.0, 9.0, 100.0],
                "d": [1, 1, 0, 0, 0, 0, 0],
                "ps": [0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.5],
                "x1": [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5],
            }
        )

    def test_all_averages_the_tied_controls(self):
        got = _fit(self._tied(), covariates=["x1"], pscore="ps", ties="all")
        assert got.estimate == pytest.approx(((10 - 2) + (20 - 7)) / 2, abs=1e-12)
        assert got.model_info["n_units_with_tied_matches"] == 2
        assert got.model_info["n_tied_matches_left_out"] == 0

    def test_first_keeps_the_first_in_index_order_and_says_so(self):
        with pytest.warns(UserWarning, match="ties='first' kept the first"):
            got = sp.match(
                self._tied(),
                y="y",
                treat="d",
                covariates=["x1"],
                pscore="ps",
                ties="first",
            )
        assert got.estimate == pytest.approx(((10 - 1) + (20 - 5)) / 2, abs=1e-12)
        assert got.model_info["n_tied_matches_left_out"] == 2

    def test_all_does_not_depend_on_the_order_of_the_rows(self):
        df = self._tied()
        base = _fit(df, covariates=["x1"], pscore="ps").estimate
        for seed in range(3):
            shuffled = df.sample(frac=1.0, random_state=seed)
            assert _fit(shuffled, covariates=["x1"], pscore="ps").estimate == (
                pytest.approx(base, abs=1e-12)
            )
            # a fresh index removes the labels the tie-break relies on
            again = shuffled.reset_index(drop=True)
            assert _fit(again, covariates=["x1"], pscore="ps").estimate == (
                pytest.approx(base, abs=1e-12)
            )

    def test_tie_tolerance_pools_near_ties(self):
        df = self._tied()
        df.loc[3, "ps"] = 0.2501  # no longer exactly tied with row 2
        strict = _fit(df, covariates=["x1"], pscore="ps")
        # scaled squared distance of the near tie is 1e-8 / var(ps) ~ 1.6e-7
        loose = _fit(df, covariates=["x1"], pscore="ps", tie_tolerance=1e-5)
        assert strict.estimate == pytest.approx(((10 - 1) + (20 - 7)) / 2, abs=1e-12)
        assert loose.estimate == pytest.approx(((10 - 2) + (20 - 7)) / 2, abs=1e-12)

    def test_mixed_type_index_labels_do_not_break_the_tie_order(self):
        df = self._tied()
        df.index = ["a", 1, "b", 2, "c", 3, "d"]
        got = _fit(df, covariates=["x1"], pscore="ps", ties="all")
        assert got.estimate == pytest.approx(((10 - 2) + (20 - 7)) / 2, abs=1e-12)


class TestWithoutReplacement:
    @staticmethod
    def _greedy(pt, pc, yt, yc, order):
        used, effects = set(), []
        for i in order:
            free = [j for j in range(len(pc)) if j not in used]
            j = min(free, key=lambda j: (abs(pt[i] - pc[j]), j))
            used.add(j)
            effects.append(yt[i] - yc[j])
        return float(np.mean(effects))

    @pytest.mark.parametrize("m_order", ["data", "smallest", "largest"])
    def test_static_orders(self, m_order):
        df = _small()
        t, c = df[df.d == 1], df[df.d == 0]
        pt, pc = t.ps.to_numpy(), c.ps.to_numpy()
        order = {
            "data": np.arange(len(pt)),
            "smallest": np.argsort(pt),
            "largest": np.argsort(-pt),
        }[m_order]
        want = self._greedy(pt, pc, t.y.to_numpy(), c.y.to_numpy(), order)
        got = _fit(df, covariates=["x1"], pscore="ps", replace=False, m_order=m_order)
        assert got.estimate == pytest.approx(want, rel=1e-12)
        # no control is used twice
        w = got.matched_data.loc[df.d == 0, "_weight"].dropna()
        assert len(w) == len(t) and (w == 1).all()

    @pytest.mark.parametrize("m_order", ["closest", "farthest", "smallest_min_dist"])
    def test_distance_driven_orders(self, m_order):
        df = _small()
        t, c = df[df.d == 1], df[df.d == 0]
        pt, pc = t.ps.to_numpy(), c.ps.to_numpy()
        yt, yc = t.y.to_numpy(), c.y.to_numpy()
        dist = np.abs(pt[:, None] - pc[None, :])
        if m_order == "smallest_min_dist":
            want = self._greedy(pt, pc, yt, yc, np.argsort(dist.min(axis=1)))
        else:
            used, left, effects = set(), set(range(len(pt))), []
            pick = min if m_order == "closest" else max
            while left:
                free = [j for j in range(len(pc)) if j not in used]
                i = pick(sorted(left), key=lambda i: dist[i, free].min())
                j = min(free, key=lambda j: (dist[i, j], j))
                used.add(j)
                left.discard(i)
                effects.append(yt[i] - yc[j])
            want = float(np.mean(effects))
        got = _fit(df, covariates=["x1"], pscore="ps", replace=False, m_order=m_order)
        assert got.estimate == pytest.approx(want, rel=1e-12)

    def test_an_exhausted_pool_is_reported(self):
        df = _small(n_t=9, n_c=5)
        with pytest.warns(UserWarning, match="pool is exhausted"):
            got = sp.match(
                df, y="y", treat="d", covariates=["x1"], pscore="ps", replace=False
            )
        assert got.model_info["n_treated_unmatched"] == 4
        assert got.model_info["n_matched_treated"] == 5


class TestKernelFamily:
    @staticmethod
    def _kernel(u, name):
        inside = np.abs(u) <= 1
        return {
            "epan": np.where(inside, 1 - u**2, 0.0),
            "biweight": np.where(inside, (1 - u**2) ** 2, 0.0),
            "uniform": inside.astype(float),
            "tricube": np.where(inside, (1 - np.abs(u) ** 3) ** 3, 0.0),
            "normal": stats.norm.pdf(u),
        }[name]

    @pytest.mark.parametrize(
        "kernel", ["epan", "biweight", "uniform", "tricube", "normal"]
    )
    def test_kernel_att_and_psmatch2_se(self, kernel):
        df = _small(n_t=8, n_c=20)
        t, c = df[df.d == 1], df[df.d == 0]
        bw = 0.15
        k = self._kernel(
            (t.ps.to_numpy()[:, None] - c.ps.to_numpy()[None, :]) / bw, kernel
        )
        keep = k.sum(axis=1) > 0
        w = k[keep] / k[keep].sum(axis=1, keepdims=True)
        want = float(np.mean(t.y.to_numpy()[keep] - w @ c.y.to_numpy()))
        got = _fit(
            df,
            covariates=["x1"],
            pscore="ps",
            method="kernel",
            kernel=kernel,
            bwidth=bw,
        )
        assert got.estimate == pytest.approx(want, rel=1e-12)
        assert got.model_info["n_matched_treated"] == int(keep.sum())

        wc = w.sum(axis=0)
        n1 = int(keep.sum())
        se = np.sqrt(
            t.y.to_numpy()[keep].var(ddof=1) / n1
            + c.y.to_numpy()[wc > 0].var(ddof=1) * np.sum(wc**2) / n1**2
        )
        assert got.se == pytest.approx(se, rel=1e-10)

    def test_radius_is_a_uniform_kernel_with_the_caliper_as_bandwidth(self):
        df = _small(n_t=8, n_c=20)
        a = _fit(df, covariates=["x1"], pscore="ps", method="radius", caliper=0.1)
        b = _fit(
            df,
            covariates=["x1"],
            pscore="ps",
            method="kernel",
            kernel="uniform",
            bwidth=0.1,
        )
        assert a.estimate == pytest.approx(b.estimate, rel=1e-12)
        assert a.se == pytest.approx(b.se, rel=1e-12)
        # n_matches has no role in kernel matching
        c = _fit(
            df,
            covariates=["x1"],
            pscore="ps",
            method="radius",
            caliper=0.1,
            n_matches=4,
        )
        assert c.estimate == a.estimate

    def test_local_linear_matching_is_exact_for_a_linear_control_mean(self):
        # The local-linear weights satisfy sum w = 1 and sum w (p_j - p_i) = 0,
        # so a control outcome that is linear in the score is reproduced
        # without error and the ATT is the constant effect.
        df = _small(n_t=8, n_c=20)
        df["y"] = 1.0 + 4.0 * df.ps + 2.5 * df.d
        got = _fit(
            df,
            covariates=["x1"],
            pscore="ps",
            method="llr",
            kernel="tricube",
            bwidth=0.5,
            bootstrap_reps=5,
            bootstrap_seed=0,
        )
        assert got.estimate == pytest.approx(2.5, abs=1e-10)
        assert got.model_info["se_method"] == "bootstrap"
        # kernel matching on the same data is biased by the slope
        k = _fit(df, covariates=["x1"], pscore="ps", method="kernel", bwidth=0.5)
        assert abs(k.estimate - 2.5) > 1e-3

    def test_no_control_inside_the_bandwidth_is_an_error(self):
        df = _small()
        with pytest.raises(DataInsufficient, match="no treated unit with a control"):
            _fit(df, covariates=["x1"], pscore="ps", method="kernel", bwidth=1e-9)

    def test_llr_needs_variation_in_the_donor_scores(self):
        # Every donor has the same score, so the local linear slope is not
        # identified for any treated unit: the documented behaviour is to
        # drop each one off support, which leaves nothing to average.
        df = _small()
        df.loc[df.d == 0, "ps"] = 0.5
        with pytest.raises(DataInsufficient):
            _fit(
                df,
                covariates=["x1"],
                pscore="ps",
                method="llr",
                kernel="tricube",
                bwidth=0.9,
                bootstrap_reps=2,
            )

    def test_kernel_common_support_removes_off_support_treated(self):
        df = _small(n_t=8, n_c=20)
        df.loc[0, "ps"] = 0.99
        got = _fit(
            df,
            covariates=["x1"],
            pscore="ps",
            method="kernel",
            kernel="normal",
            bwidth=0.2,
            common_support="minmax",
        )
        t, c = df[df.d == 1].iloc[1:], df[df.d == 0]
        k = stats.norm.pdf((t.ps.to_numpy()[:, None] - c.ps.to_numpy()[None, :]) / 0.2)
        w = k / k.sum(axis=1, keepdims=True)
        assert got.estimate == pytest.approx(
            float(np.mean(t.y.to_numpy() - w @ c.y.to_numpy())), rel=1e-12
        )

    def test_bootstrap_se_is_the_sd_of_refits_on_arm_stratified_resamples(self):
        df = _data(n=150)
        reps, seed = 6, 3
        got = _fit(
            df,
            method="kernel",
            bwidth=0.2,
            se_method="bootstrap",
            bootstrap_reps=reps,
            bootstrap_seed=seed,
        )
        rng = np.random.default_rng(seed)
        t_rows, c_rows = np.where(df.d == 1)[0], np.where(df.d == 0)[0]
        draws = []
        for _ in range(reps):
            take = np.r_[
                rng.choice(t_rows, len(t_rows), replace=True),
                rng.choice(c_rows, len(c_rows), replace=True),
            ]
            rep = _fit(
                df.iloc[take].reset_index(drop=True), method="kernel", bwidth=0.2
            )
            draws.append(rep.estimate)
        assert got.se == pytest.approx(np.std(draws, ddof=1), rel=1e-10)
        assert got.model_info["bootstrap_reps_successful"] == reps
        assert got.model_info["bootstrap_bias"] == pytest.approx(
            np.mean(draws) - got.estimate, abs=1e-10
        )

    def test_a_single_bootstrap_replication_is_not_a_standard_error(self):
        df = _data(n=150)
        with pytest.warns(UserWarning, match="only 1 usable replication"):
            sp.match(
                df,
                y="y",
                treat="d",
                covariates=X,
                method="kernel",
                bwidth=0.2,
                se_method="bootstrap",
                bootstrap_reps=1,
                bootstrap_seed=0,
            )

    def test_bootstrap_for_nearest_neighbour_matching_warns(self):
        df = _data(n=120)
        with pytest.warns(UserWarning, match="Abadie & Imbens 2008"):
            sp.match(
                df,
                y="y",
                treat="d",
                covariates=X,
                se_method="bootstrap",
                bootstrap_reps=3,
                bootstrap_seed=0,
            )


class TestStratifyCemExact:
    def test_stratification_on_score_quantiles(self):
        # n - 1 = 58 is not a multiple of 5, so no observation sits on an
        # interior quantile and the right-open bins of the estimator and
        # the right-closed bins of pd.qcut hold the same units.
        df = _small(seed=4, n_t=24, n_c=35)
        s = pd.qcut(df.ps, 5, labels=False)
        rows = []
        for _, g in df.groupby(s):
            gt, gc = g[g.d == 1], g[g.d == 0]
            if len(gt) and len(gc):
                v = (gt.y.var(ddof=1) / len(gt) if len(gt) > 1 else 0.0) + (
                    gc.y.var(ddof=1) / len(gc) if len(gc) > 1 else 0.0
                )
                rows.append((gt.y.mean() - gc.y.mean(), len(gt), len(g), v))
        tau, n_t, n_all, var = map(np.asarray, zip(*rows))
        for estimand, w in (("ATT", n_t), ("ATE", n_all)):
            w = w / w.sum()
            got = _fit(
                df, covariates=["x1"], pscore="ps", method="stratify", estimand=estimand
            )
            assert got.estimate == pytest.approx(float(tau @ w), rel=1e-12)
            assert got.se == pytest.approx(float(np.sqrt(w**2 @ var)), rel=1e-12)
            assert got.model_info["n_effective_strata"] == len(rows)

        given = _fit(
            df.assign(block=s),
            covariates=["x1"],
            pscore="ps",
            method="stratify",
            strata="block",
        )
        assert given.estimate == pytest.approx(
            _fit(df, covariates=["x1"], pscore="ps", method="stratify").estimate,
            rel=1e-12,
        )
        assert given.model_info["strata_source"] == "given"

    def test_given_strata_with_missing_labels_and_a_one_arm_stratum(self):
        df = pd.DataFrame(
            {
                "y": [5.0, 7.0, 1.0, 2.0, 9.0, 4.0, 50.0, 60.0],
                "d": [1, 1, 0, 0, 1, 0, 1, 0],
                "x1": [0.1, 0.4, 0.2, 0.5, 0.9, 0.8, 0.3, 0.6],
                "block": ["a", "a", "a", "a", "b", "b", "c", np.nan],
            }
        )
        # block c has no control once the row with the missing label is
        # left out, so only a (2 treated) and b (1 treated) contribute.
        got = _fit(df, covariates=["x1"], method="stratify", strata="block")
        assert got.estimate == pytest.approx((2 * (6 - 1.5) + 1 * (9 - 4)) / 3)
        assert got.n_obs == 7
        assert got.model_info["n_treated_in_strata"] == 3
        assert got.model_info["n_control_in_strata"] == 3

    def test_cem_att_from_the_cells(self):
        df = _data(n=200)
        bins = 4
        cell = [
            pd.cut(
                df[v],
                np.linspace(df[v].min(), df[v].max(), bins + 1),
                include_lowest=True,
                labels=False,
            )
            for v in X
        ]
        num = den = 0.0
        n_t = n_c = 0
        for _, g in df.groupby(cell):
            gt, gc = g[g.d == 1], g[g.d == 0]
            if len(gt) and len(gc):
                num += len(gt) * (gt.y.mean() - gc.y.mean())
                den += len(gt)
                n_t, n_c = n_t + len(gt), n_c + len(gc)
        got = _fit(df, method="cem", n_bins=bins)
        assert got.estimate == pytest.approx(num / den, rel=1e-12)
        assert got.model_info["n_matched_treated"] == n_t
        assert got.model_info["n_matched_control"] == n_c
        assert got.model_info["n_unmatched_treated"] == int(df.d.sum()) - n_t
        assert _fit(df, method="cem", n_bins=[bins, bins]).estimate == got.estimate
        assert (
            _fit(df, method="cem", n_bins={"x1": bins, "x2": bins}).estimate
            == got.estimate
        )

    def test_cem_with_cut_edges(self):
        df = _data(n=200)
        edges = [-0.5, 0.5]
        # values beyond the outer edges form cells of their own
        cell = np.searchsorted(edges, df.x1.to_numpy(), side="left")
        cell[df.x1.to_numpy() < edges[0]] = -1
        num = den = 0.0
        for _, g in df.groupby(cell):
            gt, gc = g[g.d == 1], g[g.d == 0]
            num += len(gt) * (gt.y.mean() - gc.y.mean())
            den += len(gt)
        got = _fit(df, covariates=["x1"], method="cem", n_bins={"x1": edges})
        assert got.estimate == pytest.approx(num / den, rel=1e-12)
        assert got.model_info["n_bins"] == {"x1": edges}

    def test_cem_without_a_common_cell_is_an_error(self):
        df = pd.DataFrame(
            {"y": [1.0, 2.0, 3.0, 4.0], "d": [1, 1, 0, 0], "x1": [0.0, 0.1, 5.0, 5.1]}
        )
        with pytest.raises(DataInsufficient, match="no strata with both"):
            _fit(df, covariates=["x1"], method="cem", n_bins=2)

    def test_exact_matching(self):
        rng = np.random.default_rng(5)
        n = 60
        df = pd.DataFrame(
            {"g": rng.integers(0, 4, n), "h": rng.integers(0, 2, n)}
        ).astype(float)
        df["d"] = rng.binomial(1, 0.4, n)
        df.loc[(df.g == 3) & (df.h == 1), "d"] = 1  # a cell with no control
        df["y"] = df.g + 2 * df.d + rng.normal(size=n)
        effects = []
        for _, g in df.groupby(["g", "h"]):
            gt, gc = g[g.d == 1], g[g.d == 0]
            if len(gc):
                effects += list(gt.y - gc.y.mean())
        got = _fit(df, covariates=["g", "h"], distance="exact")
        assert got.estimate == pytest.approx(np.mean(effects), rel=1e-12)
        assert got.se == pytest.approx(
            np.std(effects, ddof=1) / np.sqrt(len(effects)), rel=1e-12
        )
        n_lost = int(((df.g == 3) & (df.h == 1)).sum())
        assert got.model_info["n_unmatched_treated"] == n_lost

    def test_exact_matching_without_any_match_is_an_error(self):
        df = _data(n=60)  # continuous covariates: no two rows agree
        with pytest.raises(DataInsufficient, match="no treated units with exact"):
            _fit(df, distance="exact")


class TestEstimandsAndIdentities:
    def test_ate_is_the_size_weighted_mix_of_att_and_atc(self):
        df = _data()
        n1, n0 = int(df.d.sum()), int((1 - df.d).sum())
        att = _fit(df).estimate
        # the ATC is the ATT of the relabelled problem with the sign turned
        atc = -_fit(df.assign(d=1 - df.d)).estimate
        ate = _fit(df, estimand="ATE")
        # the relabelled logit gives 1 - p up to rounding, so the matched
        # pairs are the same and only the last digits can move
        assert ate.estimate == pytest.approx(
            (n1 * att + n0 * atc) / (n1 + n0), rel=1e-9
        )
        assert ate.estimand == "ATE"
        assert ate.model_info["matched_frame_weight_kind"] == "ate_signed"
        md = ate.matched_data
        # the frame documents: ATE = mean((2 treated - 1) * weight * y)
        assert float(np.mean((2 * md.d - 1) * md["_weight"] * md.y)) == pytest.approx(
            ate.estimate, rel=1e-9
        )

    def test_relabelling_the_arms_turns_the_sign_of_the_ate(self):
        df = _data()
        a = _fit(df, estimand="ATE", se_method="abadie_imbens_2016")
        b = _fit(df.assign(d=1 - df.d), estimand="ATE", se_method="abadie_imbens_2016")
        assert b.estimate == pytest.approx(-a.estimate, rel=1e-9)
        # the Abadie-Imbens (2016) ATE variance treats the arms alike
        assert b.se == pytest.approx(a.se, rel=1e-7)

    def test_a_score_given_as_a_column_reproduces_the_internal_fit(self):
        df = _data()
        inside = _fit(df)
        given = _fit(df.assign(ps=inside.matched_data["_pscore"]), pscore="ps")
        assert given.estimate == pytest.approx(inside.estimate, rel=1e-12)
        assert given.se == pytest.approx(inside.se, rel=1e-12)

    def test_rows_with_a_missing_score_are_left_out(self):
        df = _data()
        ps = _fit(df).matched_data["_pscore"].to_numpy().copy()
        ps[:10] = np.nan
        got = _fit(df.assign(ps=ps), pscore="ps")
        want = _fit(df.assign(ps=ps).iloc[10:], pscore="ps")
        assert got.n_obs == len(df) - 10
        assert got.estimate == pytest.approx(want.estimate, rel=1e-12)

    @pytest.mark.parametrize("distance", ["propensity", "euclidean", "mahalanobis"])
    def test_a_constant_covariate_changes_nothing(self, distance):
        df = _data(n=150)
        base = _fit(df, distance=distance)
        with_c = _fit(df.assign(c=3.0), covariates=X + ["c"], distance=distance)
        assert with_c.estimate == pytest.approx(base.estimate, rel=1e-9)

    def test_missing_covariate_rows_are_dropped(self):
        df = _data(n=150)
        df.loc[[3, 40], "x1"] = np.nan
        got = _fit(df)
        want = _fit(df.dropna())
        assert got.n_obs == 148
        assert got.estimate == pytest.approx(want.estimate, rel=1e-12)
        # the frame keeps every input row; the dropped ones carry no score
        assert len(got.matched_data) == 150
        assert got.matched_data.loc[[3, 40], "_pscore"].isna().all()

    def test_categorical_covariate_is_expanded_to_indicators(self):
        df = _data(n=200)
        rng = np.random.default_rng(1)
        df["g"] = rng.choice(["a", "b", "c"], len(df))
        dummies = df.assign(gb=(df.g == "b") * 1.0, gc=(df.g == "c") * 1.0)
        got = _fit(df, covariates=X + ["g"], distance="mahalanobis")
        want = _fit(dummies, covariates=X + ["gb", "gc"], distance="mahalanobis")
        assert got.estimate == pytest.approx(want.estimate, rel=1e-12)

    def test_bias_correction_removes_a_linear_discrepancy_exactly(self):
        # With a control mean that is exactly linear in the covariates the
        # regression fitted on the controls is exact, so the corrected
        # estimate is the constant effect whatever the matching discrepancy.
        df = _data(n=150)
        df["y"] = 1.0 + 2.0 * df.x1 - 1.5 * df.x2 + 0.75 * df.d
        raw = _fit(df, distance="mahalanobis")
        bc = _fit(df, distance="mahalanobis", bias_correction=True)
        assert bc.estimate == pytest.approx(0.75, abs=1e-10)
        assert abs(raw.estimate - 0.75) > 1e-3
        assert bc.method == "Matching (Mahalanobis, BC)"
        ate = _fit(df, distance="mahalanobis", bias_correction=True, estimand="ATE")
        assert ate.estimate == pytest.approx(0.75, abs=1e-10)

    def test_interval_and_p_value_follow_from_estimate_and_se(self):
        df = _data(n=150)
        got = _fit(df, alpha=0.1)
        z = stats.norm.ppf(0.95)
        assert got.ci[0] == pytest.approx(got.estimate - z * got.se, rel=1e-12)
        assert got.ci[1] == pytest.approx(got.estimate + z * got.se, rel=1e-12)
        assert got.pvalue == pytest.approx(
            2 * stats.norm.sf(abs(got.estimate / got.se)), rel=1e-10
        )

    def test_probit_and_polynomial_scores(self):
        sm = pytest.importorskip("statsmodels.api")
        df = _data(n=200)
        probit = sm.Probit(df.d, sm.add_constant(df[X])).fit(disp=0).predict()
        got = _fit(df, ps_model="probit")
        # two maximisers of the same likelihood agree to optimiser tolerance
        np.testing.assert_allclose(got.matched_data["_pscore"], probit, atol=1e-6)

        quad = df.assign(a=df.x1**2, b=df.x1 * df.x2, c=df.x2**2)
        logit = (
            sm.Logit(df.d, sm.add_constant(quad[X + ["a", "b", "c"]]))
            .fit(disp=0)
            .predict()
        )
        got2 = _fit(df, ps_poly=2)
        np.testing.assert_allclose(got2.matched_data["_pscore"], logit, atol=1e-6)


class TestFrequencyWeights:
    def test_integer_weights_are_an_expansion_of_the_rows(self):
        df = _data(n=120)
        rng = np.random.default_rng(2)
        df["w"] = rng.integers(1, 4, len(df))
        expanded = df.loc[df.index.repeat(df.w)].reset_index(drop=True)
        got = _fit(df, weights="w", method="kernel", bwidth=0.2)
        want = _fit(expanded, method="kernel", bwidth=0.2)
        assert got.estimate == pytest.approx(want.estimate, rel=1e-12)
        assert got.n_obs == int(df.w.sum())
        assert got.model_info["frequency_weights"] == "w"
        assert got.model_info["n_rows_before_expansion"] == len(df)

    def test_unit_weights_change_nothing(self):
        df = _data(n=120)
        a, b = _fit(df), _fit(df.assign(w=1), weights="w")
        assert b.estimate == pytest.approx(a.estimate, rel=1e-12)
        assert b.se == pytest.approx(a.se, rel=1e-12)

    def test_rows_with_a_missing_weight_are_left_out(self):
        df = _data(n=120)
        w = np.ones(len(df))
        w[:6] = np.nan
        got = _fit(df.assign(w=w), weights="w")
        assert got.n_obs == len(df) - 6
        assert got.estimate == pytest.approx(_fit(df.iloc[6:]).estimate, rel=1e-12)

    @pytest.mark.parametrize("bad", [1.5, 0.0, -1.0])
    def test_sampling_weights_are_refused(self, bad):
        df = _data(n=60)
        with pytest.raises(MethodIncompatibility, match="frequency weights"):
            _fit(df.assign(w=bad), weights="w")

    def test_unknown_weight_column(self):
        with pytest.raises(MethodIncompatibility, match="weights column 'nope'"):
            _fit(_data(n=60), weights="nope")

    def test_cluster_is_refused_with_alternatives(self):
        with pytest.raises(MethodIncompatibility, match="cluster-robust matching"):
            sp.match(_data(n=60), y="y", treat="d", covariates=X, cluster="x1")

    def test_doubling_every_row_shrinks_the_se_by_root_two(self):
        # Frequency weight 2 on every row is the same estimate from twice
        # the information. In the Abadie-Imbens variance
        # sum_i (1 + K_i)^2 sigma_i^2 / N1^2 the matching weights K_i are
        # unchanged per copy, the number of terms doubles and N1^2
        # quadruples, so the variance halves. The tolerance is wide because
        # sigma_i^2 is re-estimated; the defect is a factor of infinity.
        df = _data(n=200)
        base = _fit(df)
        doubled = _fit(df.assign(w=2), weights="w")
        assert doubled.estimate == pytest.approx(base.estimate, rel=1e-12)
        assert doubled.se == pytest.approx(base.se / np.sqrt(2), rel=0.3)
        assert doubled.pvalue < 1e-6


class TestBalanceOutput:
    def test_detail_is_the_balance_before_matching(self):
        df = _data(n=150)
        got = _fit(df)
        bal = got.detail.set_index("variable")
        for v in X:
            t, c = df.loc[df.d == 1, v], df.loc[df.d == 0, v]
            smd = (t.mean() - c.mean()) / np.sqrt((t.var() + c.var()) / 2)
            # the table is rounded to four decimals
            assert bal.loc[v, "smd"] == pytest.approx(smd, abs=5e-5)
            assert bal.loc[v, "mean_treated"] == pytest.approx(t.mean(), abs=5e-5)
        assert "propensity_score" in bal.index

    def test_balanceplot_and_psplot_draw_the_numbers_they_report(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = _data(n=150)
        got = _fit(df)
        fig, ax = sp.balanceplot(got, threshold=0.2, title="t")
        drawn = np.asarray(ax.collections[0].get_offsets())[:, 0]
        np.testing.assert_allclose(drawn, got.detail["smd"].to_numpy())
        assert [t.get_text() for t in ax.get_yticklabels()] == list(
            got.detail["variable"]
        )
        assert ax.get_title() == "t"
        plt.close(fig)

        fig2, ax2 = plt.subplots()
        out_fig, out_ax = sp.balanceplot(got, ax=ax2)
        assert out_ax is ax2 and out_fig is fig2
        plt.close(fig2)

        bare = _fit(df)
        bare.detail = None
        with pytest.raises(ValueError, match="No balance table"):
            sp.balanceplot(bare)

        fig3, ax3 = sp.psplot(df, treat="d", covariates=X, n_bins=10, trim=0.1)
        # two histograms of ten bars each; each integrates to one
        heights = np.array([p.get_height() for p in ax3.patches])
        assert len(heights) == 20
        assert heights[:10].sum() * 0.1 == pytest.approx(1.0)
        assert heights[10:].sum() * 0.1 == pytest.approx(1.0)
        assert sorted(line.get_xdata()[0] for line in ax3.lines) == [0.1, 0.9]
        plt.close(fig3)
        fig4, ax4 = plt.subplots()
        assert sp.psplot(df, treat="d", covariates=X, ax=ax4)[1] is ax4
        plt.close(fig4)


class TestValidation:
    @pytest.mark.parametrize(
        "kw, exc, msg",
        [
            ({"estimand": "ATC"}, MethodIncompatibility, "estimand must be"),
            ({"method": "nope"}, MethodIncompatibility, "method must be one of"),
            ({"distance": "nope"}, MethodIncompatibility, "distance must be one of"),
            ({"common_support": "nope"}, MethodIncompatibility, "common_support"),
            ({"se_method": "nope"}, MethodIncompatibility, "se_method must be"),
            ({"ps_model": "cloglog"}, MethodIncompatibility, "ps_model must be"),
            ({"caliper_scale": "nope"}, ValueError, "caliper_scale"),
            ({"ties": "nope"}, ValueError, "ties must be"),
            ({"tie_tolerance": -1.0}, ValueError, "tie_tolerance"),
            ({"m_order": "nope"}, ValueError, "m_order must be"),
            ({"mahalanobis_cov": "nope"}, ValueError, "mahalanobis_cov"),
            ({"n_matches": 0}, MethodIncompatibility, "n_matches must be a positive"),
            (
                {"n_matches": True},
                MethodIncompatibility,
                "n_matches must be a positive",
            ),
            ({"n_matches": 1.5}, MethodIncompatibility, "n_matches must be a positive"),
            ({"ai_matches": 0}, MethodIncompatibility, "ai_matches"),
            ({"bootstrap_reps": 0}, MethodIncompatibility, "bootstrap_reps"),
            ({"n_strata": 0}, MethodIncompatibility, "n_strata"),
            ({"ps_poly": 0}, MethodIncompatibility, "ps_poly"),
            ({"alpha": 1.0}, MethodIncompatibility, r"open interval \(0, 1\)"),
            ({"alpha": "a"}, MethodIncompatibility, r"open interval \(0, 1\)"),
            ({"caliper": -0.1}, MethodIncompatibility, "caliper must be finite"),
            ({"caliper": "a"}, MethodIncompatibility, "caliper must be finite"),
            ({"bwidth": 0.0}, MethodIncompatibility, "bwidth must be finite"),
            ({"bwidth": np.inf}, MethodIncompatibility, "bwidth must be finite"),
            (
                {"distance": "exact", "estimand": "ATE"},
                MethodIncompatibility,
                "exact matching only supports",
            ),
            (
                {"method": "stratify", "distance": "mahalanobis"},
                MethodIncompatibility,
                "requires distance='propensity'",
            ),
            (
                {"method": "kernel", "distance": "euclidean"},
                MethodIncompatibility,
                "requires distance='propensity'",
            ),
            (
                {"method": "kernel", "estimand": "ATE"},
                MethodIncompatibility,
                "estimand='ATT' only",
            ),
            (
                {"method": "kernel", "kernel": "nope"},
                MethodIncompatibility,
                "kernel must be one of",
            ),
            ({"method": "radius"}, MethodIncompatibility, "requires caliper"),
            ({"strata": "x1"}, MethodIncompatibility, "strata= names the strata"),
            (
                {"llr_stata_compat": True},
                MethodIncompatibility,
                "only applies to method='llr'",
            ),
            (
                {"method": "llr", "se_method": "psmatch2"},
                MethodIncompatibility,
                "not defined for method='llr'",
            ),
            (
                {"estimand": "ATE", "se_method": "psmatch2"},
                MethodIncompatibility,
                "only defined for estimand='ATT'",
            ),
            (
                {"se_method": "abadie_imbens_2016", "distance": "mahalanobis"},
                MethodIncompatibility,
                "needs 'propensity'",
            ),
            (
                {"se_method": "abadie_imbens_2016", "bias_correction": True},
                MethodIncompatibility,
                "bias_correction=True",
            ),
            (
                {"se_method": "abadie_imbens_2016", "estimand": "ATE", "ties": "first"},
                MethodIncompatibility,
                "the ATE needs 'all'",
            ),
            (
                {
                    "se_method": "abadie_imbens_2016",
                    "estimand": "ATE",
                    "replace": False,
                },
                MethodIncompatibility,
                "replace=False",
            ),
            (
                {"method": "cem", "n_bins": {"zz": 3}},
                MethodIncompatibility,
                "not covariates",
            ),
            (
                {"method": "cem", "n_bins": {"x1": [1.0, 0.0]}},
                MethodIncompatibility,
                "increasing cut edges",
            ),
            (
                {"method": "cem", "n_bins": [3]},
                MethodIncompatibility,
                "1 entries for 2 covariates",
            ),
            ({"method": "cem", "n_bins": 0}, MethodIncompatibility, "n_bins"),
            (
                {"replace": False, "m_order": "closest", "caliper": 1e-12},
                None,
                None,
            ),
        ],
    )
    def test_option_errors(self, kw, exc, msg):
        df = _data(n=60)
        if exc is None:
            # nothing within the caliper: the dynamic order ends cleanly
            # rather than looping, and no matched pair at all is an error
            # (it used to come back as an effect of 0.0)
            with pytest.raises(DataInsufficient, match="no unit found a match"):
                _fit(df, **kw)
            return
        with pytest.raises(exc, match=msg):
            _fit(df, **kw)

    def test_data_and_column_errors(self):
        df = _data(n=60)
        with pytest.raises(MethodIncompatibility, match="must be a pandas DataFrame"):
            sp.match(df.to_numpy(), y="y", treat="d", covariates=X)
        with pytest.raises(MethodIncompatibility, match="columns not found"):
            _fit(df, covariates=["x1", "zz"])
        with pytest.raises(MethodIncompatibility, match="column name or a list"):
            _fit(df, covariates=5)
        with pytest.raises(MethodIncompatibility, match="only column-name strings"):
            _fit(df, covariates=["x1", 2])
        with pytest.raises(MethodIncompatibility, match="must be binary"):
            _fit(df.assign(d=np.arange(len(df)) % 3))
        with pytest.raises(DataInsufficient, match="both treated and control"):
            _fit(df.assign(d=1))
        with pytest.raises(MethodIncompatibility, match="not a numeric column"):
            _fit(df.assign(ps="a"), pscore="ps")
        with pytest.raises(MethodIncompatibility, match="no score is estimated"):
            _fit(df.assign(ps=0.5), pscore="ps", se_method="abadie_imbens_2016")
        # a single covariate may be named without a list
        one = _fit(df, covariates="x1")
        assert one.estimate == _fit(df, covariates=["x1"]).estimate

    def test_orders_on_the_score_need_a_score(self):
        df = _small()
        # 'smallest' and 'largest' order the treated by the score; with the
        # score given as a column they must agree with sorting it by hand,
        # which TestWithoutReplacement checks. Here: they differ from each
        # other on this sample only through that order.
        a = _fit(df, covariates=["x1"], pscore="ps", replace=False, m_order="smallest")
        b = _fit(df, covariates=["x1"], pscore="ps", replace=False, m_order="largest")
        used_a = set(a.matched_data.loc[df.d == 0, "_weight"].dropna().index)
        used_b = set(b.matched_data.loc[df.d == 0, "_weight"].dropna().index)
        assert len(used_a) == len(used_b) == int(df.d.sum())


class TestRemainingBranches:
    def test_cubic_score_model(self):
        sm = pytest.importorskip("statsmodels.api")
        df = _data(n=300)
        terms = df.assign(
            a=df.x1**2, b=df.x2**2, c=df.x1 * df.x2, e=df.x1**3, f=df.x2**3
        )
        ref = (
            sm.Logit(df.d, sm.add_constant(terms[X + ["a", "b", "c", "e", "f"]]))
            .fit(disp=0)
            .predict()
        )
        got = _fit(df, ps_poly=3)
        # two maximisers of the same likelihood
        np.testing.assert_allclose(got.matched_data["_pscore"], ref, atol=1e-6)

    def test_more_argument_errors(self):
        df = _data(n=80)
        with pytest.raises(MethodIncompatibility, match="columns not found in data"):
            sp.match(df, y="zz", treat="d", covariates=X)
        with pytest.raises(MethodIncompatibility, match="needs 'nearest'"):
            _fit(df, method="kernel", se_method="abadie_imbens_2016")
        # a caliper some units meet and some do not: the 2016 variance
        # needs every unit to have its matches (a caliper nobody meets is
        # DataInsufficient before any variance is computed)
        with pytest.raises(MethodIncompatibility, match="too few propensity-score"):
            _fit(df, estimand="ATE", se_method="abadie_imbens_2016", caliper=0.01)
        with pytest.raises(DataInsufficient, match="no unit found a match"):
            _fit(df, estimand="ATE", se_method="abadie_imbens_2016", caliper=1e-6)
        with pytest.raises(DataInsufficient, match="no strata contain both"):
            _fit(df.assign(s=df.d), method="stratify", strata="s")

    def test_first_tie_warning_counts_both_arms_under_the_ate(self):
        df = _data(n=120)
        df["ps"] = np.round(1 / (1 + np.exp(-(0.5 * df.x1 - 0.3 * df.x2))), 1)
        with pytest.warns(UserWarning, match=f"of {len(df)} matched units"):
            got = sp.match(
                df,
                y="y",
                treat="d",
                covariates=X,
                pscore="ps",
                ties="first",
                estimand="ATE",
            )
        assert got.model_info["n_tied_matches_left_out"] > 0

    def test_failed_bootstrap_replications_are_counted_and_reported(self):
        # One treated unit whose only donor inside the bandwidth is row 1:
        # a resample without row 1 has nothing to match and must be counted
        # as failed, not entered as an estimate.
        df = pd.DataFrame(
            {
                "y": [5.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                "d": [1] + [0] * 10,
                "ps": [0.5, 0.5005] + list(np.linspace(0.1, 0.4, 9)),
                "x1": np.arange(11.0),
            }
        )
        reps, seed = 30, 0
        rng = np.random.default_rng(seed)
        failed = 0
        for _ in range(reps):
            rng.choice(np.array([0]), size=1, replace=True)
            failed += 1 not in rng.choice(np.arange(1, 11), size=10, replace=True)
        assert 3 < failed < reps
        with pytest.warns(UserWarning, match="bootstrap replications failed"):
            got = sp.match(
                df,
                y="y",
                treat="d",
                covariates=["x1"],
                pscore="ps",
                method="kernel",
                bwidth=0.01,
                se_method="bootstrap",
                bootstrap_reps=reps,
                bootstrap_seed=seed,
            )
        assert got.estimate == pytest.approx(4.0)
        assert got.model_info["bootstrap_reps_failed"] == failed
        assert got.model_info["bootstrap_reps_successful"] == reps - failed


# ----------------------------------------------------------------------
# Defects. Each test states the correct behaviour.
# ----------------------------------------------------------------------


class TestDefects:
    def test_ate_standard_error_does_not_depend_on_which_arm_is_called_treated(self):
        # Relabelling the arms turns the ATE into its negative: the same
        # matched pairs, the same quantity, hence the same standard error.
        # se_method='abadie_imbens_2016' satisfies this (see above).
        df = _data()
        a = _fit(df, estimand="ATE")
        b = _fit(df.assign(d=1 - df.d), estimand="ATE")
        assert b.estimate == pytest.approx(-a.estimate, rel=1e-9)
        assert b.se == pytest.approx(a.se, rel=1e-6)

    @pytest.mark.parametrize("se_method", ["abadie_imbens", "bootstrap"])
    def test_ate_honours_or_refuses_the_requested_se_method(self, se_method):
        df = _data(n=150)
        try:
            got = _fit(
                df,
                estimand="ATE",
                se_method=se_method,
                bootstrap_reps=5,
                bootstrap_seed=0,
            )
        except MethodIncompatibility:
            return  # refusing is a correct answer
        assert got.model_info.get("se_method") == se_method

    def test_cem_ate_is_weighted_by_cell_size(self):
        df = _data(n=200)
        bins = 4
        cell = [
            pd.cut(
                df[v],
                np.linspace(df[v].min(), df[v].max(), bins + 1),
                include_lowest=True,
                labels=False,
            )
            for v in X
        ]
        num = den = 0.0
        for _, g in df.groupby(cell):
            gt, gc = g[g.d == 1], g[g.d == 0]
            if len(gt) and len(gc):
                num += len(g) * (gt.y.mean() - gc.y.mean())
                den += len(g)
        try:
            got = _fit(df, method="cem", n_bins=bins, estimand="ATE")
        except MethodIncompatibility:
            return  # refusing, as distance='exact' does, is also correct
        assert got.estimate == pytest.approx(num / den, rel=1e-10)

    @pytest.mark.parametrize("estimand", ["ATT", "ATE"])
    def test_no_matched_pair_at_all_is_an_error(self, estimand):
        df = _data(n=150)
        with pytest.raises(DataInsufficient):
            _fit(df, caliper=1e-12, estimand=estimand)

    def test_no_treated_unit_on_support_is_an_error(self):
        df = _small()
        df.loc[df.d == 1, "ps"] = 0.99
        with pytest.raises(DataInsufficient):
            _fit(df, covariates=["x1"], pscore="ps", common_support="minmax")

    def test_mahalanobis_with_a_missing_covariate_value(self):
        df = _data(n=150)
        df.loc[3, "x1"] = np.nan
        got = _fit(df, distance="mahalanobis")
        want = _fit(df.dropna(), distance="mahalanobis")
        assert got.estimate == pytest.approx(want.estimate, rel=1e-12)

    def test_mahalanobis_ignores_a_redundant_linear_combination(self):
        # With the generalised inverse the Mahalanobis distance is invariant
        # to adding a column that is a linear function of the others:
        # (x-y)A (A'SA)^+ A'(x-y)' = (x-y) S^-1 (x-y)'. Brute force with
        # np.linalg.pinv on this sample gives the two-covariate estimate.
        df = _data()
        base = _fit(df, distance="mahalanobis")
        extra = _fit(
            df.assign(c=2 * df.x1 + 1), covariates=X + ["c"], distance="mahalanobis"
        )
        assert extra.estimate == pytest.approx(base.estimate, rel=1e-9)

    def test_sd_caliper_needs_a_score_distance(self):
        df = _data(n=150)
        with pytest.raises(MethodIncompatibility, match="caliper_scale='sd'"):
            _fit(df, distance="mahalanobis", caliper=0.5, caliper_scale="sd")

    def test_undefined_standard_error_has_no_p_value(self):
        df = _data()
        one = pd.concat([df[df.d == 0], df[df.d == 1].iloc[:1]])
        got = _fit(one, se_method="abadie_imbens_2016")
        assert np.isnan(got.se)  # holds today; the p-value is the defect
        assert np.isnan(got.pvalue)

    @pytest.mark.parametrize(
        "kw",
        [
            {"distance": "exact"},
            {"method": "kernel", "bwidth": 0.3},
            {"se_method": "ai"},
            {"se_method": "psmatch2"},
        ],
        ids=["exact", "kernel", "ai", "psmatch2"],
    )
    def test_single_matched_unit_has_no_standard_error(self, kw):
        df = _data()
        df["g"] = (df.x1 > 0) * 1.0
        one = pd.concat([df[df.d == 0], df[df.d == 1].iloc[:1]])
        cov = ["g"] if kw.get("distance") == "exact" else X
        got = _fit(one, covariates=cov, **kw)
        assert np.isnan(got.se)
        assert np.isnan(got.pvalue)

    def test_stratification_on_pairs_has_no_within_stratum_variance(self):
        df = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0],
                "d": [1, 0, 1, 0, 1, 0],
                "g": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            }
        )
        got = _fit(df.assign(s=df.g), covariates=["g"], method="stratify", strata="s")
        assert got.estimate == pytest.approx(-4 / 3)
        assert np.isnan(got.se)
        assert np.isnan(got.pvalue)

    def test_ate_with_a_binding_caliper_says_that_units_were_dropped(self):
        df = _data()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sp.match(df, y="y", treat="d", covariates=X, estimand="ATE", caliper=0.002)
        text = " ".join(str(w.message) for w in caught)
        # 48 of 153 treated and 38 of 147 controls have no match here
        assert "no match" in text or "EXCLUDED" in text

    @pytest.mark.parametrize("method", ["stratify", "cem"])
    def test_cell_methods_honour_or_refuse_se_method(self, method):
        df = _data(n=150)
        try:
            got = _fit(
                df,
                method=method,
                se_method="bootstrap",
                bootstrap_reps=5,
                bootstrap_seed=0,
            )
        except MethodIncompatibility:
            return
        assert got.model_info.get("se_method") == "bootstrap"

    def test_ate_under_minmax_support_is_refused_or_uses_all_units(self):
        df = _data()
        try:
            trimmed = _fit(df, estimand="ATE", common_support="minmax")
        except MethodIncompatibility:
            return
        assert trimmed.estimate == pytest.approx(
            _fit(df, estimand="ATE").estimate, rel=1e-12
        )


# ----------------------------------------------------------------------
# The fixes: what replaced the defects above, against independent numbers.
# ----------------------------------------------------------------------


class TestAteStandardErrors:
    # R 4.x, Matching 4.10-15, on _data() (seed 0, n = 300), matching on
    # the fitted logit score written out to 17 digits:
    #   Match(Y, Tr, X = ps, estimand = "ATE", M = k, ties = TRUE,
    #         sample = TRUE,  Var.calc = 1)$se   -> "sample"
    #   Match(..., sample = FALSE, Var.calc = 0)$se -> "pop"
    # and with X = cbind(x1, x2), Weight = 2 for the Mahalanobis rows.
    R_SCORE = {
        (1, "abadie_imbens"): (1.773112233484, 0.239943109380),
        (2, "abadie_imbens"): (1.913366708919, 0.214298262944),
        (1, "abadie_imbens_pop"): (1.773112233484, 0.257904840476),
        (2, "abadie_imbens_pop"): (1.913366708919, 0.237280404284),
    }
    R_MAHALANOBIS = {
        (1, "abadie_imbens"): (2.129687649244, 0.133681301472),
        (2, "abadie_imbens"): (2.052249243077, 0.132683802384),
        (1, "abadie_imbens_pop"): (2.129687649244, 0.163484492477),
        (2, "abadie_imbens_pop"): (2.052249243077, 0.155788969685),
    }

    @pytest.mark.parametrize("key", sorted(R_SCORE))
    def test_score_matching_against_r_matching(self, key):
        k, se_method = key
        got = _fit(_data(), estimand="ATE", n_matches=k, se_method=se_method)
        est, se = self.R_SCORE[key]
        # R prints 12 decimals
        assert got.estimate == pytest.approx(est, abs=2e-12)
        assert got.se == pytest.approx(se, abs=2e-12)
        assert got.model_info["se_method"] == se_method

    @pytest.mark.parametrize("key", sorted(R_MAHALANOBIS))
    def test_mahalanobis_matching_against_r_matching(self, key):
        k, se_method = key
        got = _fit(
            _data(),
            estimand="ATE",
            n_matches=k,
            se_method=se_method,
            distance="mahalanobis",
            mahalanobis_cov="total",  # Matching's Weight = 2
        )
        est, se = self.R_MAHALANOBIS[key]
        assert got.estimate == pytest.approx(est, abs=2e-12)
        assert got.se == pytest.approx(se, abs=2e-12)

    def test_the_default_is_the_abadie_imbens_variance(self):
        df = _data()
        auto = _fit(df, estimand="ATE")
        assert auto.model_info["se_method"] == "abadie_imbens"
        assert auto.se == _fit(df, estimand="ATE", se_method="abadie_imbens").se

    @pytest.mark.parametrize(
        "se_method", ["auto", "ai", "abadie_imbens", "abadie_imbens_pop"]
    )
    @pytest.mark.parametrize("distance", ["propensity", "mahalanobis"])
    def test_every_analytic_option_is_symmetric_in_the_arms(self, se_method, distance):
        df = _data()
        kw = dict(estimand="ATE", se_method=se_method, distance=distance)
        a = _fit(df, **kw)
        b = _fit(df.assign(d=1 - df.d), **kw)
        assert b.estimate == pytest.approx(-a.estimate, rel=1e-9)
        assert b.se == pytest.approx(a.se, rel=1e-9)

    def test_ai_combines_the_two_legs(self):
        # Var = [n_t^2 Var(mean_t) + n_c^2 Var(mean_c)] / N^2 with the
        # matched-pair variance of each leg, written out by brute force.
        df = _small()
        t, c = df[df.d == 1], df[df.d == 0]
        dist = np.abs(t.ps.to_numpy()[:, None] - c.ps.to_numpy()[None, :])
        _, e_t = _nn_att(t.y.to_numpy(), c.y.to_numpy(), dist)
        _, e_c = _nn_att(c.y.to_numpy(), t.y.to_numpy(), dist.T)
        n_t, n_c = len(t), len(c)
        want = np.sqrt(
            n_t**2 * e_t.var(ddof=1) / n_t + n_c**2 * e_c.var(ddof=1) / n_c
        ) / (n_t + n_c)
        got = _fit(df, covariates=["x1"], pscore="ps", estimand="ATE", se_method="ai")
        assert got.se == pytest.approx(want, rel=1e-12)

    def test_population_variance_needs_every_unit_matched(self):
        df = _data()
        with pytest.warns(UserWarning, match="needs every unit to have a match"):
            got = sp.match(
                df,
                y="y",
                treat="d",
                covariates=X,
                estimand="ATE",
                caliper=0.002,
                se_method="abadie_imbens_pop",
            )
        assert np.isnan(got.se) and np.isnan(got.pvalue)
        assert np.all(np.isnan(got.ci))
        assert got.model_info["n_treated_unmatched"] > 0
        assert got.model_info["n_control_unmatched"] > 0

    def test_caliper_variance_follows_the_estimate(self):
        # With units dropped by the caliper the estimate is still linear in
        # the outcomes; the coefficients the variance uses reproduce it.
        from statspai.matching.match import MatchEstimator

        df = _data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = MatchEstimator(
                df, "y", "d", X, estimand="ATE", caliper=0.002, se_method="ai"
            )
            res = est.fit()
        coef = est._ate_coefficients(est._assignment, *est._ate_reverse)
        assert float(coef @ df.y.to_numpy()) == pytest.approx(res.estimate, rel=1e-12)


class TestFrequencyWeightVariance:
    def test_doubling_is_exactly_root_two(self):
        # each copy keeps its matching weight and its sigma^2 (taken from
        # another observation), the number of terms doubles, N1^2 quadruples
        df = _data(n=200)
        for kw in ({}, {"estimand": "ATE"}, {"distance": "mahalanobis"}):
            base = _fit(df, **kw)
            doubled = _fit(df.assign(w=2), weights="w", **kw)
            assert doubled.se == pytest.approx(base.se / np.sqrt(2), rel=1e-10)

    def test_neighbours_of_other_observations_only(self):
        from statspai.matching._matched_frame import (
            _self_outcome_distinct_origin,
            _within_group_self_outcome,
        )

        rng = np.random.default_rng(3)
        n = 60
        t = rng.integers(0, 2, n)
        ps = np.round(rng.uniform(size=n), 1)  # many exact ties
        y = rng.normal(size=n)
        for j in (1, 2, 4):
            # one row per observation: the ordinary neighbours
            np.testing.assert_allclose(
                _self_outcome_distinct_origin(y, t, ps, j, np.arange(n)),
                _within_group_self_outcome(y, t, ps, j),
                rtol=1e-12,
            )
        # three copies of each row: a copy is never its own neighbour, and
        # with J = 3 the neighbours are the three copies of the nearest
        # other observation, whose mean is that observation's outcome
        rep = np.repeat(np.arange(n), 3)
        got = _self_outcome_distinct_origin(y[rep], t[rep], ps[rep], 3, rep)
        want = _within_group_self_outcome(y, t, ps, 1)[rep]
        np.testing.assert_allclose(got, want, rtol=1e-12)


class TestUndefinedInference:
    def test_an_outcome_without_variation_has_no_test(self):
        df = _data(n=120)
        df["y"] = 3.0 + 2.0 * df.d  # every matched effect is exactly 2
        with pytest.warns(UserWarning, match="zero up to rounding"):
            got = sp.match(df, y="y", treat="d", covariates=X)
        assert got.estimate == pytest.approx(2.0)
        assert got.se <= 1e-10 * 5.0
        assert np.isnan(got.pvalue)

    def test_a_stratum_with_one_unit_of_an_arm_leaves_the_se_undefined(self):
        df = pd.DataFrame(
            {
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 6.0, 9.0],
                "d": [1, 1, 0, 0, 1, 0, 0, 0],
                "g": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
            }
        )
        with pytest.warns(UserWarning, match="1 of 2 strata hold a single"):
            got = sp.match(
                df.assign(s=df.g),
                y="y",
                treat="d",
                covariates=["g"],
                method="stratify",
                strata="s",
            )
        assert got.estimate == pytest.approx((2 * (1.5 - 3.5) + (5 - 22 / 3)) / 3)
        assert np.isnan(got.se)

    def test_cem_ate_standard_error_is_the_stratified_one(self):
        df = _data(n=400)
        bins = 2
        cell = [
            pd.cut(
                df[v],
                np.linspace(df[v].min(), df[v].max(), bins + 1),
                include_lowest=True,
                labels=False,
            )
            for v in X
        ]
        var = den = 0.0
        for _, g in df.groupby(cell):
            gt, gc = g[g.d == 1], g[g.d == 0]
            assert len(gt) > 1 and len(gc) > 1
            var += len(g) ** 2 * (gt.y.var() / len(gt) + gc.y.var() / len(gc))
            den += len(g)
        got = _fit(df, method="cem", n_bins=bins, estimand="ATE")
        assert got.se == pytest.approx(np.sqrt(var) / den, rel=1e-12)
        assert got.model_info["se_method"] == "analytic"


class TestRefusals:
    @pytest.mark.parametrize(
        "kw, msg",
        [
            ({"replace": "no"}, "replace must be True or False"),
            ({"ps_poly": 4}, "ps_poly must be 1, 2 or 3"),
            ({"method": "stratify", "se_method": "abadie_imbens"}, "not defined for"),
            ({"method": "cem", "se_method": "ai"}, "not defined for"),
            (
                {"distance": "exact", "se_method": "abadie_imbens"},
                "not defined for distance='exact'",
            ),
        ],
    )
    def test_options_that_used_to_be_ignored(self, kw, msg):
        df = _data(n=80)
        df["g"] = (df.x1 > 0) * 1.0
        cov = ["g"] if kw.get("distance") == "exact" else X
        with pytest.raises(MethodIncompatibility, match=msg):
            _fit(df, covariates=cov, **kw)

    def test_replace_accepts_zero_and_one(self):
        df = _data(n=80)
        assert _fit(df, replace=0).estimate == _fit(df, replace=False).estimate


class TestMahalanobisInverse:
    def test_full_rank_is_the_ordinary_inverse(self):
        from statspai.matching.match import _mahalanobis_inverse

        rng = np.random.default_rng(0)
        cov = np.cov(rng.normal(size=(50, 3)).T * np.array([[1.0], [1e3], [1e-3]]))
        np.testing.assert_array_equal(_mahalanobis_inverse(cov), np.linalg.inv(cov))

    def test_a_constant_covariate_adds_no_distance(self):
        df = _data()
        base = _fit(df, distance="mahalanobis")
        extra = _fit(df.assign(c=7.0), covariates=X + ["c"], distance="mahalanobis")
        assert extra.estimate == pytest.approx(base.estimate, rel=1e-9)


class TestCovariateDistanceVarianceAgainstRMatching:
    # R 4.5.2, Matching 4.10-15, on _data() (seed 0, n = 300), with
    # X = cbind(x1, x2), ties = TRUE, replace = TRUE, distance.tolerance = 0:
    #   Match(..., estimand = e, M = k, Weight = W, sample = TRUE,
    #         Var.calc = 1)$se                      -> "abadie_imbens"
    #   Match(..., sample = FALSE, Var.calc = 0)$se -> "abadie_imbens_pop"
    # Weight = 2 is the Mahalanobis distance on the full-sample covariance
    # (mahalanobis_cov='total'); Weight = 1 scales each covariate by its
    # standard deviation (distance='euclidean').
    # (distance, estimand, M): (estimate, sample SE, population SE)
    R = {
        ("mahalanobis", "ATT", 1): (2.235040249115, 0.167775904190, 0.201539451118),
        ("mahalanobis", "ATT", 2): (2.220817899852, 0.139933857162, 0.182197096979),
        ("mahalanobis", "ATE", 1): (2.129687649244, 0.133681301472, 0.163484492477),
        ("mahalanobis", "ATE", 2): (2.052249243077, 0.132683802384, 0.155788969685),
        ("euclidean", "ATT", 1): (2.252848194976, 0.172925531818, 0.202459766273),
        ("euclidean", "ATT", 2): (2.232429813832, 0.142275461796, 0.182677889234),
        ("euclidean", "ATE", 1): (2.136439639864, 0.137278439825, 0.163503746892),
        ("euclidean", "ATE", 2): (2.055380841555, 0.134567678564, 0.155225029266),
    }

    @pytest.mark.parametrize("key", sorted(R))
    def test_sample_and_population_variance(self, key):
        distance, estimand, k = key
        kw = dict(distance=distance, estimand=estimand, n_matches=k)
        if distance == "mahalanobis":
            kw["mahalanobis_cov"] = "total"
        est, se_sample, se_pop = self.R[key]
        sample = _fit(_data(), se_method="abadie_imbens", **kw)
        pop = _fit(_data(), se_method="abadie_imbens_pop", **kw)
        # R prints 12 decimals
        assert sample.estimate == pytest.approx(est, abs=2e-12)
        assert sample.se == pytest.approx(se_sample, abs=2e-12)
        assert pop.se == pytest.approx(se_pop, abs=2e-12)

    @pytest.mark.parametrize("distance", ["mahalanobis", "euclidean"])
    def test_the_att_neighbours_are_found_in_the_matching_metric(self, distance):
        # sigma^2(X_i) = (Y_i - Y_l)^2 / 2 with l the nearest same-arm unit
        # in the matching distance, and Var = sum_i c_i^2 sigma^2(X_i) / N1^2
        # with c_i = 1 for a treated unit and the matching weight K_i for a
        # control: written out with a full distance matrix.
        df = _data(n=120)
        got = _fit(df, distance=distance, se_method="abadie_imbens")
        x, y, d = df[X].to_numpy(), df.y.to_numpy(), df.d.to_numpy()
        if distance == "euclidean":
            full = cdist(x / x.std(axis=0, ddof=1), x / x.std(axis=0, ddof=1))
        else:
            s1, s0 = np.cov(x[d == 1].T), np.cov(x[d == 0].T)
            n1, n0 = (d == 1).sum(), (d == 0).sum()
            pooled = ((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2)
            full = cdist(x, x, metric="mahalanobis", VI=np.linalg.inv(pooled))
        same = d[:, None] == d[None, :]
        within = np.where(same, full, np.inf)
        np.fill_diagonal(within, np.inf)
        sigma2 = (y - y[within.argmin(axis=1)]) ** 2 / 2
        across = np.where(~same, full, np.inf)
        coef = (d == 1).astype(float)
        for i in np.flatnonzero(d == 1):
            coef[across[i].argmin()] += 1.0
        want = np.sqrt(np.sum(coef**2 * sigma2)) / (d == 1).sum()
        assert got.se == pytest.approx(want, rel=1e-12)

    def test_the_score_distance_keeps_its_neighbours_on_the_score(self):
        # psmatch2's ai(): there the score is the matching metric
        from statspai.matching._matched_frame import abadie_imbens_se

        df = _data(n=120)
        got = _fit(df, se_method="abadie_imbens")
        md = got.matched_data
        want = abadie_imbens_se(
            df.y.to_numpy(),
            df.d.to_numpy(),
            md["_pscore"].to_numpy(),
            md["_support"].to_numpy(),
            md["_weight"].to_numpy(),
        )
        assert got.se == want
