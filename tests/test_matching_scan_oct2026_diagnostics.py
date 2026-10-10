"""Propensity-score diagnostics: balance tables, trimming, overlap and love plots.

Covers the branches of ``matching/ps_diagnostics.py`` the suite did not reach.
Every statistic is recomputed in the test from its definition; plots are
checked through the data they draw.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.matching.ps_diagnostics import (
    _crump_alpha,
    _group_variance,
    _ks_stat,
    _smd,
    _variance_ratio,
)

X = ["x1", "x2"]


def _data(seed: int = 0, n: int = 300, strength: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-strength * (0.5 * x1 - 0.3 * x2))))
    y = 1 + 2 * d + x1 + 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


def _wvar(x, w):
    m = np.average(x, weights=w)
    return np.sum(w * (x - m) ** 2) / (w.sum() - np.sum(w**2) / w.sum())


def _wsmd(xt, xc, wt, wc):
    return (np.average(xt, weights=wt) - np.average(xc, weights=wc)) / np.sqrt(
        (_wvar(xt, wt) + _wvar(xc, wc)) / 2
    )


class TestPropensityScore:
    def test_logit_and_probit_are_the_maximum_likelihood_fits(self):
        sm = pytest.importorskip("statsmodels.api")
        df = _data()
        design = sm.add_constant(df[X])
        logit = sm.Logit(df.d, design).fit(disp=0).predict()
        probit = sm.Probit(df.d, design).fit(disp=0).predict()
        got = sp.propensity_score(df, "d", X)
        # two Newton iterations on the same likelihood
        np.testing.assert_allclose(got, logit, atol=1e-10)
        assert got.name == "propensity_score" and got.index.equals(df.index)
        # BFGS stops at its gradient tolerance, hence the looser bound
        np.testing.assert_allclose(
            sp.propensity_score(df, "d", X, method="probit"), probit, atol=1e-5
        )

    def test_gbm_is_the_documented_sklearn_model(self):
        ensemble = pytest.importorskip("sklearn.ensemble")
        df = _data(n=200)
        ref = ensemble.GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        ).fit(df[X].to_numpy(), df.d.to_numpy(dtype=float))
        want = np.clip(ref.predict_proba(df[X].to_numpy())[:, 1], 1e-8, 1 - 1e-8)
        got = sp.propensity_score(df, "d", X, method="gbm")
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="'logit', 'probit', or 'gbm'"):
            sp.propensity_score(_data(n=50), "d", X, method="nope")

    def test_crump_cutoff_solves_its_fixed_point(self):
        # gamma = 2 E[g | g <= gamma] with g = 1 / (e (1 - e)), and the
        # cutoff alpha satisfies alpha (1 - alpha) = 1 / gamma.
        df = _data(strength=6.0)
        ps = sp.propensity_score(df, "d", X).to_numpy()
        alpha = _crump_alpha(ps)
        assert 0 < alpha < 0.5
        gamma = 1 / (alpha * (1 - alpha))
        g = 1 / (ps * (1 - ps))
        assert 2 * g[g <= gamma].mean() == pytest.approx(gamma, rel=1e-10)

        trimmed = sp.propensity_score(df, "d", X, trimming="crump")
        out = (ps < alpha) | (ps > 1 - alpha)
        assert out.sum() > 0
        assert trimmed.isna().to_numpy().tolist() == out.tolist()

    def test_crump_cutoff_degenerate_inputs(self):
        # scores close to one half: max g <= 2 mean g, nothing is trimmed
        assert _crump_alpha(np.array([0.45, 0.5, 0.55, 0.5])) == 0.0
        assert _crump_alpha(np.array([0.3])) == 0.0
        assert _crump_alpha(np.array([np.nan, 0.0, 1.0, 0.4])) == 0.0

    def test_trimming_keeps_the_rows_inside_the_cutoffs(self):
        df = _data(strength=6.0)
        ps = sp.propensity_score(df, "d", X)
        alpha = _crump_alpha(ps.to_numpy())
        crump = sp.trimming(df, treatment="d", covariates=X)
        assert crump.index.tolist() == (
            df.index[(ps >= alpha) & (ps <= 1 - alpha)].tolist()
        )
        sturmer = sp.trimming(df, treatment="d", covariates=X, method="sturmer")
        assert sturmer.index.tolist() == df.index[(ps >= 0.1) & (ps <= 0.9)].tolist()
        # a score supplied by the caller is used as it is; missing scores go
        given = ps.copy()
        given.iloc[:5] = np.nan
        own = sp.trimming(df, treatment="d", covariates=X, ps=given, method="sturmer")
        assert own.index.tolist() == [i for i in sturmer.index if i >= 5]
        with pytest.raises(ValueError, match="'crump' or 'sturmer'"):
            sp.trimming(df, treatment="d", covariates=X, method="nope")
        # treat= is an accepted spelling
        assert len(sp.trimming(df, treat="d", covariates=X)) == len(crump)


class TestBalanceStatistics:
    def test_weighted_variance_special_cases(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=40)
        # equal weights of any size give the n - 1 variance
        assert _group_variance(x, np.full(40, 3.7)) == pytest.approx(
            x.var(ddof=1), rel=1e-12
        )
        # 0/1 weights give the variance of the retained rows
        keep = (rng.uniform(size=40) < 0.5).astype(float)
        assert _group_variance(x, keep) == pytest.approx(
            x[keep == 1].var(ddof=1), rel=1e-12
        )
        assert np.isnan(_group_variance(x, np.zeros(40)))
        assert _group_variance(np.array([2.0]), None) == 0.0
        # a single retained row has no n - 1 variance: the sum of squares
        # over the total weight (zero) is returned
        one = np.zeros(40)
        one[3] = 2.0
        assert _group_variance(x, one) == pytest.approx(0.0, abs=1e-15)

    def test_smd_denominators(self):
        rng = np.random.default_rng(1)
        xt, xc = rng.normal(0.3, 1, 30), rng.normal(0, 2, 50)
        wt, wc = rng.uniform(0.5, 2, 30), rng.uniform(0.5, 2, 50)
        assert _smd(xt, xc) == pytest.approx(
            (xt.mean() - xc.mean()) / np.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2),
            rel=1e-12,
        )
        assert _smd(xt, xc, wt, wc) == pytest.approx(_wsmd(xt, xc, wt, wc), rel=1e-12)
        unweighted_sd = np.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2)
        assert _smd(xt, xc, wt, wc, sd_denom="unweighted") == pytest.approx(
            (np.average(xt, weights=wt) - np.average(xc, weights=wc)) / unweighted_sd,
            rel=1e-12,
        )
        # an undefined statistic is not a balanced one
        assert np.isnan(_smd(np.r_[xt, np.nan], xc))
        # identical constants in both groups: no difference, no spread
        assert _smd(np.ones(5), np.ones(7)) == 0.0

    def test_variance_ratio(self):
        rng = np.random.default_rng(2)
        xt, xc = rng.normal(0, 2, 30), rng.normal(0, 1, 50)
        assert _variance_ratio(xt, xc) == pytest.approx(
            xt.var(ddof=1) / xc.var(ddof=1), rel=1e-12
        )
        assert _variance_ratio(xt, np.ones(50)) == np.inf
        assert _variance_ratio(np.ones(30), np.ones(50)) == 1.0

    def test_weighted_ks_is_the_largest_gap_between_step_functions(self):
        rng = np.random.default_rng(3)
        xt, xc = rng.normal(0.4, 1, 25), rng.normal(0, 1, 35)
        wt, wc = rng.uniform(0.2, 3, 25), rng.uniform(0.2, 3, 35)
        gaps = [
            abs(wt[xt <= v].sum() / wt.sum() - wc[xc <= v].sum() / wc.sum())
            for v in np.r_[xt, xc]
        ]
        assert _ks_stat(xt, xc, wt, wc) == pytest.approx(max(gaps), rel=1e-12)
        # weights on one side only
        gaps_c = [
            abs(np.mean(xt <= v) - wc[xc <= v].sum() / wc.sum()) for v in np.r_[xt, xc]
        ]
        assert _ks_stat(xt, xc, None, wc) == pytest.approx(max(gaps_c), rel=1e-12)
        gaps_t = [
            abs(wt[xt <= v].sum() / wt.sum() - np.mean(xc <= v)) for v in np.r_[xt, xc]
        ]
        assert _ks_stat(xt, xc, wt, None) == pytest.approx(max(gaps_t), rel=1e-12)
        # unit weights reproduce the unweighted statistic
        assert _ks_stat(xt, xc, np.ones(25), np.ones(35)) == pytest.approx(
            stats.ks_2samp(xt, xc).statistic, rel=1e-12
        )
        assert np.isnan(_ks_stat(xt, xc, np.zeros(25), wc))
        assert _ks_stat(xt, xc) == stats.ks_2samp(xt, xc).statistic


class TestPsBalance:
    def test_default_weights_are_inverse_probability_weights(self):
        df = _data()
        res = sp.ps_balance(df, treatment="d", covariates=X)
        ps = sp.propensity_score(df, "d", X).to_numpy()
        w = np.where(df.d == 1, 1 / ps, 1 / (1 - ps))
        t, c = (df.d == 1).to_numpy(), (df.d == 0).to_numpy()
        for v in X:
            x = df[v].to_numpy()
            row = res.table.loc[v]
            assert row["mean_treat"] == pytest.approx(x[t].mean(), rel=1e-12)
            assert row["smd_weighted"] == pytest.approx(
                _wsmd(x[t], x[c], w[t], w[c]), rel=1e-10
            )
            assert row["variance_ratio"] == pytest.approx(
                _wvar(x[t], w[t]) / _wvar(x[c], w[c]), rel=1e-10
            )
        np.testing.assert_allclose(res.ps, ps)

    def test_supplied_weights_and_denominator(self):
        df = _data()
        rng = np.random.default_rng(4)
        w = rng.uniform(0.2, 2.0, len(df))
        w[:7] = np.nan  # a missing weight is an unmatched row
        res = sp.ps_balance(
            df, treatment="d", covariates=X, weights=w, sd_denom="unweighted"
        )
        w0 = np.nan_to_num(w)
        t, c = (df.d == 1).to_numpy(), (df.d == 0).to_numpy()
        x = df.x1.to_numpy()
        want = (
            np.average(x[t], weights=w0[t]) - np.average(x[c], weights=w0[c])
        ) / np.sqrt((x[t].var(ddof=1) + x[c].var(ddof=1)) / 2)
        assert res.table.loc["x1", "smd_weighted"] == pytest.approx(want, rel=1e-12)

    def test_weight_and_option_errors(self):
        df = _data(n=60)
        with pytest.raises(ValueError, match="finite and non-negative"):
            sp.ps_balance(df, treatment="d", covariates=X, weights=-np.ones(60))
        with pytest.raises(ValueError, match="finite and non-negative"):
            sp.ps_balance(df, treatment="d", covariates=X, weights=np.full(60, np.inf))
        with pytest.raises(ValueError, match="control group has no positive weight"):
            sp.ps_balance(
                df, treatment="d", covariates=X, weights=df.d.to_numpy(dtype=float)
            )
        with pytest.raises(MethodIncompatibility, match="sd_denom must be"):
            sp.ps_balance(df, treatment="d", covariates=X, sd_denom="nope")

    def test_summary_counts_the_imbalanced_covariates(self):
        df = _data()
        res = sp.ps_balance(df, treatment="d", covariates=X)
        n_raw = int((res.table["smd_raw"].abs() > 0.1).sum())
        n_w = int((res.table["smd_weighted"].abs() > 0.1).sum())
        line = f"Covariates with |SMD| > 0.1: {n_raw} (raw) -> {n_w} (weighted)"
        assert line in res.summary()
        assert repr(res) == res.summary()
        html = res._repr_html_()
        assert f"{n_raw} (raw)" in html and "<table" in html


class TestBalanceDiagnostics:
    def test_table_and_summary_statistics(self):
        df = _data()
        rng = np.random.default_rng(5)
        df["w"] = rng.uniform(0.2, 2.0, len(df))
        df["e"] = np.clip(rng.uniform(size=len(df)), 0.05, 0.95)
        res = sp.balance_diagnostics(
            df, treatment="d", covariates=X, weights="w", ps="e", threshold=0.05
        )
        w = df.w.to_numpy()
        t, c = (df.d == 1).to_numpy(), (df.d == 0).to_numpy()
        for v in X:
            x = df[v].to_numpy()
            smd = _wsmd(x[t], x[c], w[t], w[c])
            row = res.table.loc[v]
            assert row["smd_weighted"] == pytest.approx(smd, rel=1e-10)
            assert row["weighted_mean_treat"] == pytest.approx(
                np.average(x[t], weights=w[t]), rel=1e-12
            )
            assert bool(row["balanced"]) == (abs(smd) <= 0.05)
        s = res.summary_stats
        assert s["effective_sample_size"] == pytest.approx(
            w.sum() ** 2 / np.sum(w**2), rel=1e-12
        )
        assert s["effective_sample_size_treated"] == pytest.approx(
            w[t].sum() ** 2 / np.sum(w[t] ** 2), rel=1e-12
        )
        e = df.e.to_numpy()
        lo, hi = max(e[t].min(), e[c].min()), min(e[t].max(), e[c].max())
        assert s["common_support_low"] == lo and s["common_support_high"] == hi
        assert s["common_support_width"] == pytest.approx(hi - lo)
        assert s["n_imbalanced_weighted"] == int(
            (res.table["smd_weighted"].abs() > 0.05).sum()
        )
        assert s["energy_distance_weighted"] == pytest.approx(
            sp.energy_distance(df, "d", X, weights=w), rel=1e-12
        )
        assert s["energy_distance_raw"] == pytest.approx(
            sp.energy_distance(df, "d", X), rel=1e-12
        )
        text = res.summary()
        assert f"Effective sample size: {s['effective_sample_size']:.2f}" in text
        assert repr(res) == text
        as_dict = res.to_dict()
        assert as_dict["summary"]["n_obs"] == len(df)
        assert set(as_dict["table"]) == set(X)

    def test_default_weights_are_the_inverse_probability_weights(self):
        df = _data()
        res = sp.balance_diagnostics(df, treatment="d", covariates=X)
        ref = sp.ps_balance(df, treatment="d", covariates=X).table
        np.testing.assert_allclose(res.table["smd_weighted"], ref["smd_weighted"])
        np.testing.assert_allclose(res.table["ks_stat_weighted"], ref["ks_stat"])
        ps = sp.propensity_score(df, "d", X).to_numpy()
        np.testing.assert_allclose(
            res.weights, np.where(df.d == 1, 1 / ps, 1 / (1 - ps)), rtol=1e-12
        )
        # treat= is an accepted spelling
        again = sp.balance_diagnostics(df, treat="d", covariates=X)
        pd.testing.assert_frame_equal(again.table, res.table)

    def test_every_way_of_passing_weights_gives_the_same_table(self):
        df = _data(n=120)
        rng = np.random.default_rng(6)
        w = rng.uniform(0.2, 2.0, len(df))
        df.loc[[4, 9], "x1"] = np.nan  # two incomplete rows
        complete = df.dropna().index
        by_name = sp.balance_diagnostics(df.assign(w=w), "d", X, weights="w")
        by_series = sp.balance_diagnostics(
            df, "d", X, weights=pd.Series(w, index=df.index)
        )
        by_full = sp.balance_diagnostics(df, "d", X, weights=w)
        by_complete = sp.balance_diagnostics(df, "d", X, weights=w[complete])
        for other in (by_series, by_full, by_complete):
            pd.testing.assert_frame_equal(by_name.table, other.table)
        assert by_name.summary_stats["n_obs"] == len(df) - 2
        assert by_name.weights.index.equals(complete)

    def test_errors(self):
        df = _data(n=60)
        with pytest.raises(ValueError, match="No complete observations"):
            sp.balance_diagnostics(df.assign(x1=np.nan), "d", X)
        with pytest.raises(ValueError, match="must be binary"):
            sp.balance_diagnostics(df.assign(d=np.arange(60) % 3), "d", X)
        with pytest.raises(ValueError, match="Column 'nope' not found for weights"):
            sp.balance_diagnostics(df, "d", X, weights="nope")
        with pytest.raises(ValueError, match="weights length must match"):
            sp.balance_diagnostics(df, "d", X, weights=np.ones(7))
        with pytest.raises(MethodIncompatibility, match="sd_denom must be"):
            sp.balance_diagnostics(df, "d", X, sd_denom="nope")

    def test_energy_distance_is_skipped_on_large_samples(self):
        df = _data(n=5001)
        res = sp.balance_diagnostics(
            df, "d", X, weights=np.ones(5001), ps=0.5 + 0 * df.y
        )
        assert np.isnan(res.summary_stats["energy_distance_raw"])
        assert np.isnan(res.summary_stats["energy_distance_weighted"])
        # unit weights: the weighted column repeats the raw one
        np.testing.assert_allclose(res.table["smd_weighted"], res.table["smd_raw"])


class TestPlots:
    @pytest.fixture(autouse=True)
    def _backend(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        yield
        plt.close("all")

    def test_overlap_plot_draws_the_two_densities_mirrored(self):
        df = _data()
        ps = sp.propensity_score(df, "d", X)
        fig, ax = sp.overlap_plot(df, treatment="d", covariates=X, title="t")
        grid = np.linspace(0, 1, 300)
        top = stats.gaussian_kde(ps[df.d == 1], bw_method="scott")(grid)
        bottom = stats.gaussian_kde(ps[df.d == 0], bw_method="scott")(grid)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), top, rtol=1e-10)
        np.testing.assert_allclose(ax.lines[1].get_ydata(), -bottom, rtol=1e-10)
        labels = ax.get_legend_handles_labels()[1]
        n1 = int(df.d.sum())
        assert f"Treated (n={n1})" in labels
        assert f"Control (n={len(df) - n1})" in labels
        lo = max(ps[df.d == 1].min(), ps[df.d == 0].min())
        hi = min(ps[df.d == 1].max(), ps[df.d == 0].max())
        assert f"Common support [{lo:.2f}, {hi:.2f}]" in labels
        assert ax.get_title() == "t"

        import matplotlib.pyplot as plt

        fig2, ax2 = plt.subplots()
        out = sp.overlap_plot(df, treatment="d", covariates=X, ps=ps, ax=ax2)
        assert out[0] is fig2 and out[1] is ax2
        np.testing.assert_allclose(ax2.lines[0].get_ydata(), top, rtol=1e-10)

    def test_love_plot_draws_the_absolute_differences(self):
        df = _data()
        bal = sp.ps_balance(df, treatment="d", covariates=X)
        fig, ax = sp.love_plot(df, treatment="d", covariates=X, threshold=0.2)
        raw, weighted = (np.asarray(c.get_offsets())[:, 0] for c in ax.collections)
        np.testing.assert_allclose(raw, bal.table["smd_raw"].abs())
        np.testing.assert_allclose(weighted, bal.table["smd_weighted"].abs())
        assert [t.get_text() for t in ax.get_yticklabels()] == X
        # the threshold is the one vertical line; the others join the dots
        assert ax.lines[-1].get_xdata()[0] == 0.2
        assert fig.get_size_inches()[1] == 3  # max(3, 0.4 * 2 + 1)

        fig_m, ax_m = bal.love_plot(threshold=0.15)
        raw_m = np.asarray(ax_m.collections[0].get_offsets())[:, 0]
        np.testing.assert_allclose(raw_m, raw)

        import matplotlib.pyplot as plt

        fig2, ax2 = plt.subplots()
        w = np.ones(len(df))
        out = sp.love_plot(df, treatment="d", covariates=X, weights=w, ax=ax2)
        assert out[1] is ax2
        raw2, w2 = (np.asarray(c.get_offsets())[:, 0] for c in ax2.collections)
        np.testing.assert_allclose(w2, raw2)  # unit weights change nothing

    @pytest.mark.parametrize(
        "kw",
        [
            {},
            {"n_matches": 3},
            {"method": "kernel", "bwidth": 0.2},
            {"method": "cem", "n_bins": 4},
            {"method": "stratify"},
        ],
        ids=["nearest", "nearest3", "kernel", "cem", "stratify"],
    )
    def test_love_plot_of_a_matching_result_uses_the_matching_weights(self, kw):
        df = _data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.match(df, y="y", treat="d", covariates=X, **kw)
        fig, ax = sp.love_plot(res)
        after = np.asarray(ax.collections[1].get_offsets())[:, 0]
        md = res.matched_data
        w = md["_weight"].fillna(0.0).to_numpy()
        t, c = (md.d == 1).to_numpy(), (md.d == 0).to_numpy()
        want = [
            abs(_wsmd(md[v].to_numpy()[t], md[v].to_numpy()[c], w[t], w[c])) for v in X
        ]
        np.testing.assert_allclose(after, want, rtol=1e-10)

    def test_love_plot_of_a_result_accepts_overrides(self):
        df = _data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sp.psmatch2(df, treat="d", covariates=X, outcome="y")
        fig, ax = sp.love_plot(m, covariates=["x2"])
        assert [t.get_text() for t in ax.get_yticklabels()] == ["x2"]
        after = np.asarray(ax.collections[1].get_offsets())[:, 0]
        assert after[0] == pytest.approx(
            abs(m.balance().table.loc["x2", "smd_weighted"]), rel=1e-10
        )

    def test_love_plot_needs_a_balance_specification(self):
        df = _data(n=60)
        with pytest.raises(MethodIncompatibility, match="requires treatment= and"):
            sp.love_plot(df)
        with pytest.raises(MethodIncompatibility, match="cannot read a balance"):
            sp.love_plot(object())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exact = sp.match(
                df.assign(g=(df.x1 > 0) * 1.0),
                y="y",
                treat="d",
                covariates=["g"],
                distance="exact",
            )
        # exact matching produces no matched frame to read weights from
        with pytest.raises(MethodIncompatibility, match="cannot read a balance"):
            sp.love_plot(exact)


# ----------------------------------------------------------------------
# Defects. Each test states the correct behaviour.
# ----------------------------------------------------------------------


class TestDefects:
    def test_a_repeated_covariate_does_not_change_the_scores(self):
        # The model is the same model; statsmodels' fit of the two-covariate
        # logit is the reference (it agrees to 1e-10 without the repeat).
        sm = pytest.importorskip("statsmodels.api")
        df = _data()
        want = sm.Logit(df.d, sm.add_constant(df[X])).fit(disp=0).predict()
        got = sp.propensity_score(df.assign(x3=df.x1), "d", X + ["x3"])
        np.testing.assert_allclose(got, want, atol=1e-6)

    def test_a_covariate_that_separates_the_arms_is_not_balanced(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "d": [1] * 5 + [0] * 5,
                "z": [1.0] * 5 + [0.0] * 5,
                "x": rng.normal(size=10),
            }
        )
        assert not np.isfinite(_smd(df.z.to_numpy()[:5], df.z.to_numpy()[5:]))
        res = sp.balance_diagnostics(
            df, "d", ["z", "x"], weights=np.ones(10), ps=np.full(10, 0.5)
        )
        assert not bool(res.table.loc["z", "balanced"])

    def test_one_missing_covariate_value_does_not_erase_every_score(self):
        df = _data()
        df.loc[5, "x1"] = np.nan
        try:
            ps = sp.propensity_score(df, "d", X)
        except (ValueError, MethodIncompatibility):
            return  # refusing missing values is a correct answer
        want = sp.propensity_score(df.dropna(), "d", X)
        np.testing.assert_allclose(ps.drop(index=5), want, atol=1e-10)

    def test_love_plot_of_a_result_fitted_on_data_with_missing_values(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = _data()
        df.loc[5, "x1"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.match(df, y="y", treat="d", covariates=X)
            fig, ax = sp.love_plot(res)
        drawn = np.ma.filled(
            np.ma.asarray(ax.collections[1].get_offsets())[:, 0].astype(float), np.nan
        )
        plt.close(fig)
        md = res.matched_data.dropna(subset=X)
        w = md["_weight"].fillna(0.0).to_numpy()
        t, c = (md.d == 1).to_numpy(), (md.d == 0).to_numpy()
        want = [
            abs(_wsmd(md[v].to_numpy()[t], md[v].to_numpy()[c], w[t], w[c])) for v in X
        ]
        np.testing.assert_allclose(drawn, want, rtol=1e-8)

    def test_overlap_plot_of_crump_trimmed_scores(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = _data(strength=6.0)
        ps = sp.propensity_score(df, "d", X, trimming="crump")
        kept = ps.notna()
        assert 0 < kept.sum() < len(df)
        fig, ax = sp.overlap_plot(df, treatment="d", covariates=X, ps=ps)
        labels = ax.get_legend_handles_labels()[1]
        plt.close(fig)
        assert f"Treated (n={int((kept & (df.d == 1)).sum())})" in labels
