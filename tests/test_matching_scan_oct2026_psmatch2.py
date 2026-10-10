"""sp.psmatch2: options, the matched frame and PSM-DID against hand computation.

Companion to ``test_matching_scan_oct2026_match.py``; covers the branches of
``matching/psmatch2.py`` the suite did not reach (ties / ate, the Becker-Ichino
options, result methods, PSM-DID weight regimes, argument errors).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

X = ["x1", "x2"]


def _data(seed: int = 0, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-(0.5 * x1 - 0.3 * x2))))
    y = 1 + 2 * d + x1 + 0.5 * x2 + d * x1 + rng.normal(size=n)
    return pd.DataFrame({"id": np.arange(n), "y": y, "d": d, "x1": x1, "x2": x2})


def _tied() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [10.0, 20.0, 1.0, 3.0, 5.0, 9.0, 100.0],
            "d": [1, 1, 0, 0, 0, 0, 0],
            "ps": [0.25, 0.75, 0.25, 0.25, 0.75, 0.75, 0.5],
        }
    )


def _small(seed: int = 1, n_t: int = 7, n_c: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_t + n_c
    d = np.r_[np.ones(n_t, dtype=int), np.zeros(n_c, dtype=int)]
    ps = np.round(rng.uniform(0.1, 0.9, n), 3) + np.arange(n) * 1e-5
    return pd.DataFrame(
        {"y": 2.0 * d + 3.0 * ps + rng.normal(size=n), "d": d, "ps": ps}
    )


def _psm(data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw.setdefault("treat", "d")
        if "pscore" not in kw:
            kw.setdefault("covariates", X)
        return sp.psmatch2(data, **kw)


class TestFrontDoor:
    def test_default_is_match_with_first_tie_and_the_psmatch2_se(self):
        df = _data()
        m = _psm(df, outcome="y")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = sp.match(
                df, y="y", treat="d", covariates=X, ties="first", se_method="psmatch2"
            )
        assert m.att == pytest.approx(ref.estimate, rel=1e-12)
        assert m.se == pytest.approx(ref.se, rel=1e-12)
        assert m.result.model_info["propensity_model"] == "logit"
        # y= is an alias of outcome=, n_matches= of neighbor=
        assert _psm(df, y="y").att == m.att
        assert _psm(df, outcome="y", n_matches=3).att == (
            _psm(df, outcome="y", neighbor=3).att
        )

    def test_ai_shorthand_selects_the_robust_se(self):
        df = _data()
        m = _psm(df, outcome="y", ai=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = sp.match(
                df,
                y="y",
                treat="d",
                covariates=X,
                ties="first",
                se_method="abadie_imbens",
                ai_matches=2,
            )
        assert m.se == pytest.approx(ref.se, rel=1e-12)
        assert m.result.model_info["se_method"] == "abadie_imbens"
        assert m.result.model_info["ai_matches"] == 2
        assert "AI-robust(2)" in str(m.summary())

    def test_simple_pair_se(self):
        df = _small()
        t, c = df[df.d == 1], df[df.d == 0]
        j = np.abs(t.ps.to_numpy()[:, None] - c.ps.to_numpy()[None, :]).argmin(axis=1)
        eff = t.y.to_numpy() - c.y.to_numpy()[j]
        m = _psm(df, outcome="y", pscore="ps", se="ai")
        assert m.att == pytest.approx(eff.mean(), rel=1e-12)
        assert m.se == pytest.approx(eff.std(ddof=1) / np.sqrt(len(eff)), rel=1e-12)
        # with pscore= and no covariates the score column stands in for them
        assert m.covariates == ["ps"]
        assert m.result.model_info["propensity_model"] == "given"

    def test_mahalanobis_method_matches_on_the_covariates(self):
        df = _data(n=150)
        m = _psm(df, outcome="y", method="mahalanobis")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = sp.match(
                df, y="y", treat="d", covariates=X, distance="mahalanobis", ties="first"
            )
        assert m.att == pytest.approx(ref.estimate, rel=1e-12)

    def test_without_an_outcome_only_the_frame_is_produced(self):
        df = _data(n=150)
        m = _psm(df)
        assert np.isnan(m.att) and np.isnan(m.se) and np.isnan(m.pvalue)
        assert "_y" not in m.matched_data.columns
        assert "__psmatch2_no_outcome__" not in m.matched_data.columns
        with_y = _psm(df, outcome="y")
        pd.testing.assert_series_equal(
            m.matched_data["_weight"], with_y.matched_data["_weight"]
        )
        assert m.result.model_info["att_defined"] is False


class TestTiesAndAte:
    def test_ties_splits_the_weight_between_equally_close_controls(self):
        m = _psm(_tied(), outcome="y", pscore="ps", ties=True)
        assert m.att == pytest.approx(((10 - 2) + (20 - 7)) / 2, abs=1e-12)
        w = m.matched_data["_weight"]
        assert list(w.iloc[2:6]) == [0.5, 0.5, 0.5, 0.5]
        assert np.isnan(w.iloc[6])
        # without the option the first control in data order is the match
        first = _psm(_tied(), outcome="y", pscore="ps")
        assert first.att == pytest.approx(((10 - 1) + (20 - 5)) / 2, abs=1e-12)

    def test_ties_agrees_with_match_ties_all(self):
        df = _data(n=200)
        # a coarse score, so that exact ties are everywhere
        df["ps"] = np.round(1 / (1 + np.exp(-(0.5 * df.x1 - 0.3 * df.x2))), 1)
        m = _psm(df, outcome="y", pscore="ps", ties=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = sp.match(df, y="y", treat="d", covariates=X, pscore="ps", ties="all")
        assert ref.model_info["n_units_with_tied_matches"] > 50
        assert m.att == pytest.approx(ref.estimate, rel=1e-12)

    def test_ate_reports_att_atu_and_their_mix(self):
        df = _small()
        t, c = df[df.d == 1], df[df.d == 0]
        dist = np.abs(t.ps.to_numpy()[:, None] - c.ps.to_numpy()[None, :])
        att = float(np.mean(t.y.to_numpy() - c.y.to_numpy()[dist.argmin(axis=1)]))
        atu = float(np.mean(t.y.to_numpy()[dist.argmin(axis=0)] - c.y.to_numpy()))
        m = _psm(df, outcome="y", pscore="ps", ate=True)
        info = m.result.model_info
        assert m.att == pytest.approx(att, rel=1e-12)
        assert info["atu"] == pytest.approx(atu, rel=1e-12)
        assert info["ate"] == pytest.approx(
            (len(t) * att + len(c) * atu) / len(df), rel=1e-12
        )
        # relabelling the arms exchanges the two and turns the signs
        sw = _psm(
            df.assign(d=1 - df.d, ps=1 - df.ps), outcome="y", pscore="ps", ate=True
        )
        assert sw.att == pytest.approx(-atu, rel=1e-12)
        assert sw.result.model_info["atu"] == pytest.approx(-att, rel=1e-12)
        assert sw.result.model_info["ate"] == pytest.approx(-info["ate"], rel=1e-12)

    def test_treated_range_support_with_ties(self):
        df = _small()
        t, c = df[df.d == 1], df[df.d == 0]
        lo, hi = t.ps.min(), t.ps.max()
        inside = c[(c.ps >= lo) & (c.ps <= hi)]
        assert 0 < len(inside) < len(c)
        dist = np.abs(t.ps.to_numpy()[:, None] - inside.ps.to_numpy()[None, :])
        want = float(np.mean(t.y.to_numpy() - inside.y.to_numpy()[dist.argmin(axis=1)]))
        m = _psm(df, outcome="y", pscore="ps", ties=True, common_support="treated")
        assert m.att == pytest.approx(want, rel=1e-12)
        outside = c.index.difference(inside.index)
        assert (m.matched_data.loc[outside, "_support"] == 0).all()

    def test_treated_range_support_with_a_given_score_filters_the_rows(self):
        df = _small()
        t = df[df.d == 1]
        keep = df[(df.ps >= t.ps.min()) & (df.ps <= t.ps.max())]
        m = _psm(df, outcome="y", pscore="ps", common_support="treated")
        ref = _psm(keep, outcome="y", pscore="ps")
        assert m.att == pytest.approx(ref.att, rel=1e-12)
        assert m.se == pytest.approx(ref.se, rel=1e-12)
        assert len(m.matched_data) == len(keep)

    def test_bootstrap_with_ties_refits_the_score_on_every_resample(self):
        df = _data(n=90).drop(columns="id")
        reps, seed = 5, 4
        m = _psm(
            df,
            outcome="y",
            ties=True,
            se="bootstrap",
            bootstrap_reps=reps,
            bootstrap_seed=seed,
        )
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(reps):
            rows = rng.integers(0, len(df), len(df))
            rep = _psm(df.iloc[rows].reset_index(drop=True), outcome="y", ties=True)
            draws.append(rep.att)
        # the refit inside the loop and a fresh call fit the same logit
        assert m.se == pytest.approx(np.std(draws, ddof=1), rel=1e-8)
        assert m.result.model_info["bootstrap_reps_used"] == reps
        assert m.result.model_info["se_method"] == "bootstrap"

    def test_one_bootstrap_replication_is_refused(self):
        with pytest.raises(MethodIncompatibility, match="fewer than two estimates"):
            _psm(
                _data(n=90),
                outcome="y",
                ties=True,
                se="bootstrap",
                bootstrap_reps=1,
                bootstrap_seed=0,
            )

    @pytest.mark.parametrize(
        "kw, msg",
        [
            ({"method": "kernel"}, "implemented for propensity-score nearest"),
            ({"neighbor": 2}, "neighbor=1 with replacement only"),
            ({"replace": False}, "neighbor=1 with replacement only"),
            ({"ai": 1}, "is not available with them"),
            ({"se": "ai"}, "is not available with them"),
        ],
    )
    def test_ties_restrictions(self, kw, msg):
        with pytest.raises(MethodIncompatibility, match=msg):
            _psm(_data(n=90), outcome="y", ties=True, **kw)


class TestRadiusPairs:
    def test_pooled_pairs_estimate_and_weights(self):
        df = _small(n_t=8, n_c=20)
        t, c = df[df.d == 1], df[df.d == 0]
        radius = 0.08
        near = np.abs(t.ps.to_numpy()[:, None] - c.ps.to_numpy()[None, :]) < radius
        matched = near.any(axis=1)
        pairs = near.sum(axis=0).astype(float)
        want = t.y.to_numpy()[matched].mean() - (pairs @ c.y.to_numpy()) / pairs.sum()
        m = _psm(
            df,
            outcome="y",
            pscore="ps",
            method="radius",
            caliper=radius,
            radius_weights="pairs",
        )
        assert m.att == pytest.approx(want, rel=1e-12)
        info = m.result.model_info
        assert info["n_treated_matched"] == int(matched.sum())
        assert info["n_control_used"] == int((pairs > 0).sum())
        w = m.matched_data.loc[df.d == 0, "_weight"].to_numpy()
        scaled = matched.sum() * pairs / pairs.sum()
        np.testing.assert_allclose(w[pairs > 0], scaled[pairs > 0], rtol=1e-12)
        assert np.isnan(w[pairs == 0]).all()
        # the default weighs every matched treated unit once instead
        plain = _psm(df, outcome="y", pscore="ps", method="radius", caliper=radius)
        per_treated = [
            t.y.to_numpy()[i] - c.y.to_numpy()[near_i].mean()
            for i, near_i in enumerate(
                np.abs(t.ps.to_numpy()[:, None] - c.ps.to_numpy()[None, :]) <= radius
            )
            if near_i.any()
        ]
        assert plain.att == pytest.approx(np.mean(per_treated), rel=1e-12)

    @pytest.mark.parametrize(
        "kw",
        [
            {"method": "neighbor"},
            {"method": "radius", "caliper": 0.1, "se": "ai"},
            {"method": "radius", "caliper": 0.1, "outcome": None},
        ],
    )
    def test_pairs_needs_radius_an_outcome_and_the_default_se(self, kw):
        kw = {"outcome": "y", **kw}
        with pytest.raises(MethodIncompatibility, match="radius_weights='pairs'"):
            _psm(_small(), pscore="ps", radius_weights="pairs", **kw)


class TestArgumentErrors:
    @pytest.mark.parametrize(
        "kw, msg",
        [
            ({"method": "spline"}, "method='spline' is not implemented"),
            ({"method": "nope"}, "method must be one of"),
            ({"se": "nope"}, "se must be 'psmatch2'"),
            ({"radius_weights": "nope"}, "radius_weights must be"),
            ({"method": "radius"}, "requires caliper"),
            ({"common_support": "treated"}, "implemented for ties=True or ate=True"),
            ({"covariates": X + ["y"]}, "also listed in covariates"),
            ({"covariates": ["zz"]}, "covariate column 'zz' not found"),
            ({"covariates": [""]}, "non-empty"),
            ({"covariates": [1]}, "column"),
            ({"outcome": "zz"}, "outcome column 'zz' not found"),
            ({"treat": "zz"}, "treat column 'zz' not found"),
        ],
    )
    def test_messages(self, kw, msg):
        kw = {"outcome": "y", **kw}
        with pytest.raises(MethodIncompatibility, match=msg):
            _psm(_data(n=60), **kw)

    def test_missing_roles_and_wrong_container(self):
        df = _data(n=60)
        with pytest.raises(MethodIncompatibility, match="requires treat= and"):
            sp.psmatch2(df, covariates=X)
        with pytest.raises(MethodIncompatibility, match="requires treat= and"):
            sp.psmatch2(df, treat="d")
        with pytest.raises(MethodIncompatibility, match="must be a pandas DataFrame"):
            sp.psmatch2(df.to_numpy(), treat="d", covariates=X)
        with pytest.raises(MethodIncompatibility, match="must be a pandas DataFrame"):
            sp.psmatch2(df.to_numpy(), treat="d", pscore="ps")
        with pytest.raises(MethodIncompatibility, match="pscore column 'ps' not found"):
            sp.psmatch2(df, treat="d", pscore="ps")
        # a single covariate may be given as a string
        one = _psm(df, outcome="y", covariates="x1")
        assert one.covariates == ["x1"]

    def test_llr_defaults_to_a_bootstrap_se(self):
        m = _psm(
            _data(n=120),
            outcome="y",
            method="llr",
            kernel="tricube",
            bwidth=0.3,
            bootstrap_reps=4,
            bootstrap_seed=0,
        )
        assert m.result.model_info["se_method"] == "bootstrap"
        assert m.result.model_info["bootstrap_reps_successful"] == 4


class TestResultMethods:
    def test_matched_sample_filters(self):
        df = _data()
        m = _psm(df, outcome="y", common_support="minmax", caliper=0.01)
        md = m.matched_data
        both = m.matched_sample()
        assert len(both) == int((md["_weight"].notna() & (md["_support"] == 1)).sum())
        assert len(m.matched_sample(drop_unmatched=False)) == int(
            (md["_support"] == 1).sum()
        )
        assert len(m.matched_sample(on_support=False)) == int(
            md["_weight"].notna().sum()
        )
        assert len(m.matched_sample(on_support=False, drop_unmatched=False)) == len(df)

    def test_balance_uses_the_matching_weights(self):
        df = _data()
        m = _psm(df, outcome="y", neighbor=2)
        md = m.matched_data
        w = md["_weight"].fillna(0.0).to_numpy()
        tab = m.balance().table

        def wvar(x, w):
            mean = np.average(x, weights=w)
            return np.sum(w * (x - mean) ** 2) / (w.sum() - np.sum(w**2) / w.sum())

        for v in X:
            x = md[v].to_numpy()
            xt, xc, wt, wc = x[md.d == 1], x[md.d == 0], w[md.d == 1], w[md.d == 0]
            raw = (xt.mean() - xc.mean()) / np.sqrt(
                (xt.var(ddof=1) + xc.var(ddof=1)) / 2
            )
            after = (np.average(xt, weights=wt) - np.average(xc, weights=wc)) / np.sqrt(
                (wvar(xt, wt) + wvar(xc, wc)) / 2
            )
            assert tab.loc[v, "smd_raw"] == pytest.approx(raw, rel=1e-12)
            assert tab.loc[v, "smd_weighted"] == pytest.approx(after, rel=1e-10)
            assert tab.loc[v, "weighted_mean_control"] == pytest.approx(
                np.average(xc, weights=wc), rel=1e-12
            )
        only = m.balance(covariates=["x2"], threshold=0.5)
        assert list(only.table.index) == ["x2"]
        assert only.summary_stats["threshold"] == 0.5

    def test_pstest_bias_columns(self):
        df = _data()
        m = _psm(df, outcome="y")
        res = m.pstest()
        md = m.matched_data
        w = md["_weight"].fillna(0.0).to_numpy()
        for v in X:
            x = md[v].to_numpy()
            xt, xc = x[md.d == 1], x[md.d == 0]
            # pstest keeps the unmatched pooled SD under both rows
            sd = np.sqrt((xt.var(ddof=1) + xc.var(ddof=1)) / 2)
            row = res.table.loc[v]
            assert row["mean_treated_unmatched"] == pytest.approx(xt.mean(), rel=1e-12)
            assert row["mean_control_matched"] == pytest.approx(
                np.average(xc, weights=w[md.d == 0]), rel=1e-12
            )
            assert row["pct_bias_unmatched"] == pytest.approx(
                100 * (xt.mean() - xc.mean()) / sd, rel=1e-10
            )
            assert row["pct_bias_matched"] == pytest.approx(
                100
                * (
                    np.average(xt, weights=w[md.d == 1])
                    - np.average(xc, weights=w[md.d == 0])
                )
                / sd,
                rel=1e-10,
            )
        text = str(res.summary())
        assert "x1" in text and "x2" in text
        assert list(m.pstest(covariates=["x1"]).table.index) == ["x1"]

    def test_text_and_html_summaries(self):
        df = _data(n=150)
        m = _psm(df, outcome="y", neighbor=2)
        text = str(m.summary())
        assert f"ATT               : {m.att:.4f}" in text
        assert "Neighbours (k)    : 2" in text
        assert repr(m) == text
        html = m._repr_html_()
        assert (
            f"{m.att:.4f}" in html and f"{m.n_matched_treated} / {m.n_treated}" in html
        )
        assert "abadie2006large" in m.cite()
        k = _psm(df, outcome="y", method="kernel", bwidth=0.1, common_support="minmax")
        ktext = str(k.summary())
        assert "Kernel            : epan  (bwidth: 0.1)" in ktext
        assert "Common support    : minmax" in ktext

    def test_psplot_draws_the_weighted_control_density(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy import stats

        df = _data()
        m = _psm(df, outcome="y")
        md = m.matched_data
        fig, ax = m.psplot(n_grid=50)
        grid = np.linspace(0, 1, 50)
        used = (md.d == 0) & md["_weight"].notna()
        want = stats.gaussian_kde(
            md.loc[used, "_pscore"], weights=md.loc[used, "_weight"]
        )(grid)
        # lines: matched treated, matched control (mirrored), raw control,
        # raw treated, zero line
        np.testing.assert_allclose(ax.lines[1].get_ydata(), -want, rtol=1e-10)
        raw = stats.gaussian_kde(md.loc[md.d == 0, "_pscore"])(grid)
        np.testing.assert_allclose(ax.lines[2].get_ydata(), -raw, rtol=1e-10)
        assert len(ax.lines) == 5
        plt.close(fig)

        fig2, ax2 = plt.subplots()
        out = m.overlap_plot(before=False, ax=ax2, title="t")
        assert out[1] is ax2 and len(ax2.lines) == 3 and ax2.get_title() == "t"
        plt.close(fig2)

    def test_psplot_skips_a_density_that_cannot_be_estimated(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # both treated units are matched to the same control, so the
        # matched-control density has one point and is not drawn
        df = pd.DataFrame(
            {
                "y": [1.0, 2.0, 0.0, 5.0, 6.0],
                "d": [1, 1, 0, 0, 0],
                "ps": [0.40, 0.42, 0.41, 0.90, 0.95],
            }
        )
        m = _psm(df, outcome="y", pscore="ps")
        fig, ax = m.psplot(before=False)
        assert len(ax.lines) == 2  # treated density and the zero line
        plt.close(fig)


class TestPsmDid:
    def _panel(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        pan = pd.DataFrame(
            [(i, t) for i in df.id for t in range(4)], columns=["id", "t"]
        ).merge(df[["id", "d", "x1"]], on="id")
        pan["post"] = (pan.t >= 2).astype(int)
        pan["yy"] = (
            pan.x1 + 0.3 * pan.t + 1.5 * pan.d * pan.post + rng.normal(size=len(pan))
        )
        return pan

    def test_weight_regimes_against_statsmodels(self):
        smf = pytest.importorskip("statsmodels.formula.api")
        df = _data(n=200)
        m = _psm(df, outcome="y")
        pan = self._panel(df)
        samp = pan.merge(m.matched_data[["id", "_weight"]].dropna(), on="id")

        aw = m.psm_did(pan, id="id", y="yy", post="post")
        ref = smf.wls("yy ~ d * post", samp, weights=samp["_weight"]).fit()
        assert aw.estimate == pytest.approx(ref.params["d:post"], rel=1e-10)
        assert aw.se == pytest.approx(ref.bse["d:post"], rel=1e-8)
        assert aw.n_obs == len(samp)
        assert aw.model_info["n_units_matched"] == samp.id.nunique()

        # a frequency weight is a replication of the row
        fw = m.psm_did(pan, id="id", y="yy", post="post", weight="fweight")
        rep = samp.loc[samp.index.repeat(samp["_weight"].astype(int))]
        ref_f = smf.ols("yy ~ d * post", rep).fit()
        assert fw.estimate == pytest.approx(ref_f.params["d:post"], rel=1e-10)
        assert fw.se == pytest.approx(ref_f.bse["d:post"], rel=1e-8)
        assert fw.n_obs == len(rep)

        un = m.psm_did(pan, id="id", y="yy", post="post", weight="none")
        ref_u = smf.ols("yy ~ d * post", samp).fit()
        assert un.estimate == pytest.approx(ref_u.params["d:post"], rel=1e-10)
        assert un.model_info["weight_column"] is None

        # post built from time and the first treated period
        built = m.psm_did(pan, id="id", y="yy", time="t", treat_time=2)
        assert built.estimate == pytest.approx(aw.estimate, rel=1e-12)
        assert built.se == pytest.approx(aw.se, rel=1e-12)

    def test_fixed_effects_absorb_the_main_effects(self):
        smf = pytest.importorskip("statsmodels.formula.api")
        df = _data(n=120)
        m = _psm(df, outcome="y")
        pan = self._panel(df)
        samp = pan.merge(m.matched_data[["id", "_weight"]].dropna(), on="id")
        got = m.psm_did(pan, id="id", y="yy", post="post", fixed_effects=["id", "t"])
        assert got.model_info["formula"] == "yy ~ _did | id + t"
        samp["_did"] = samp.d * samp.post
        ref = smf.wls("yy ~ _did + C(id) + C(t)", samp, weights=samp["_weight"]).fit()
        assert got.estimate == pytest.approx(ref.params["_did"], rel=1e-8)

        cl = m.psm_did(pan, id="id", y="yy", post="post", cluster=["id"])
        cl1 = m.psm_did(pan, id="id", y="yy", post="post", cluster="id")
        assert cl.se == pytest.approx(cl1.se, rel=1e-12)
        with_x = m.psm_did(pan, id="id", y="yy", post="post", covariates=["x1"])
        assert "x1" in with_x.model_info["formula"]

    def test_internal_column_names_avoid_the_panel(self):
        df = _data(n=120)
        m = _psm(df, outcome="y")
        pan = self._panel(df)
        clash = pan.assign(__statspai_psm_weight__=7.0, __statspai_psm_support__=7.0)
        a = m.psm_did(pan, id="id", y="yy", post="post")
        b = m.psm_did(clash, id="id", y="yy", post="post")
        assert b.estimate == pytest.approx(a.estimate, rel=1e-12)
        assert b.model_info["weight_column"] == "__statspai_psm_weight___1"

    def test_errors(self):
        df = _data(n=120)
        m = _psm(df, outcome="y")
        pan = self._panel(df)
        with pytest.raises(MethodIncompatibility, match="Provide either post="):
            m.psm_did(pan, id="id", y="yy")
        with pytest.raises(MethodIncompatibility, match="weight must be one of"):
            m.psm_did(pan, id="id", y="yy", post="post", weight="nope")
        with pytest.raises(MethodIncompatibility, match="cluster column 'zz'"):
            m.psm_did(pan, id="id", y="yy", post="post", cluster="zz")
        with pytest.raises(MethodIncompatibility, match="cluster column 'zz'"):
            m.psm_did(pan, id="id", y="yy", post="post", cluster=["id", "zz"])
        with pytest.raises(MethodIncompatibility, match="covariate column 'zz'"):
            m.psm_did(pan, id="id", y="yy", post="post", covariates=["zz"])
        with pytest.raises(MethodIncompatibility, match="fixed effect column 'zz'"):
            m.psm_did(pan, id="id", y="yy", post="post", fixed_effects=["zz"])
        with pytest.raises(MethodIncompatibility, match="must be a pandas DataFrame"):
            m.psm_did(pan.to_numpy(), id="id", y="yy", post="post")
        with pytest.raises(DataInsufficient, match="No matched panel rows"):
            m.psm_did(pan.assign(id=pan.id + 10_000), id="id", y="yy", post="post")
        no_id = _psm(df.drop(columns="id"), outcome="y")
        with pytest.raises(MethodIncompatibility, match="not found in the matching"):
            no_id.psm_did(pan, id="id", y="yy", post="post")
        # fractional weights cannot be frequency weights
        k2 = _psm(df, outcome="y", neighbor=2)
        with pytest.raises(MethodIncompatibility):
            k2.psm_did(pan, id="id", y="yy", post="post", weight="fweight")


class TestDefects:
    def test_no_match_within_the_caliper_is_not_a_zero_effect(self):
        df = _data(n=150)
        try:
            m = _psm(df, outcome="y", caliper=1e-12)
        except DataInsufficient:
            return
        assert m.n_matched_treated == 0  # holds today
        assert np.isnan(m.att)
        assert np.isnan(m.pvalue)

    def test_no_match_at_all_is_an_error_without_an_outcome_too(self):
        # the matched frame would carry no weight for anybody
        with pytest.raises(DataInsufficient, match="no unit found a match"):
            _psm(_data(n=150), caliper=1e-12)

    def test_no_outcome_does_not_warn_about_its_constant_placeholder(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            m = sp.psmatch2(_data(n=150), treat="d", covariates=X)
        assert np.isnan(m.att) and np.isnan(m.pvalue)

    def test_pstest_without_covariates_says_so(self):
        df = _data(n=150)
        df["ps"] = sp.propensity_score(df, "d", X)
        m = sp.psmatch2(df, treat="d", covariates=[], pscore="ps", outcome="y")
        with pytest.raises(MethodIncompatibility, match="no covariates to test"):
            m.pstest()
        assert list(m.pstest(covariates=["x1"]).table.index) == ["x1"]
