"""Option and degenerate-input tests for the smaller panel tools (scan
round 3): ``sp.fama_macbeth``, ``sp.xtsum``, ``sp.xtserial``, ``sp.xtoverid``,
``sp.xthtaylor``, ``sp.panel_unitroot``, ``sp.panel_logit`` and
``sp.interactive_fe``.

Numerical assertions compare with a computation written here, with
statsmodels (same convention, stated at the comparison) or with an identity.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility, StatsPAIError

# closed-form comparisons on small well-conditioned designs
RTOL = 1e-9


def _panel(seed: int = 0, n_units: int = 40, n_periods: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_units), n_periods)
    t = np.tile(np.arange(n_periods), n_units)
    a = rng.normal(size=n_units)[ids]
    n = n_units * n_periods
    df = pd.DataFrame(
        {
            "id": ids,
            "t": t,
            "x1": rng.normal(size=n) + a,
            "x2": rng.normal(size=n),
            "g": ids % 7,
            "z": rng.normal(size=n_units)[ids],
            "z2": (rng.normal(size=n_units) + a[::n_periods])[ids],
        }
    )
    df["y"] = 1 + 2 * df.x1 - df.x2 + a + 0.5 * df.z + 0.3 * df.z2 + rng.normal(size=n)
    df["b"] = (0.8 * df.x1 - 0.5 * df.x2 + a + rng.logistic(size=n) > 0).astype(int)
    return df


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


# ---------------------------------------------------------------- Fama-MacBeth


class TestFamaMacBeth:
    @staticmethod
    def _period_coefs(df):
        rows = []
        for _, g in df.groupby("t"):
            X = np.column_stack([np.ones(len(g)), g.x1, g.x2])
            rows.append(np.linalg.lstsq(X, g.y.to_numpy(), rcond=None)[0])
        return np.vstack(rows)

    def test_mean_of_period_regressions_and_its_time_series_se(self):
        df = _panel().drop(index=[3, 50, 51, 120])
        B = self._period_coefs(df)
        T = len(B)
        res = sp.fama_macbeth("y ~ x1 + x2", df, time="t")
        np.testing.assert_allclose(res.params.to_numpy(), B.mean(axis=0), rtol=RTOL)
        np.testing.assert_allclose(
            res.std_errors.to_numpy(), B.std(axis=0, ddof=1) / np.sqrt(T), rtol=RTOL
        )
        assert res.data_info["df_resid"] == T - 1

    def test_newey_west_lags(self):
        """Bartlett weights 1 - j/(L+1) on the autocovariances of the
        coefficient series, with the T/(T-1) factor the docstring states."""
        df = _panel()
        B = self._period_coefs(df)
        T, L = len(B), 2
        dev = B - B.mean(axis=0)
        for col in range(B.shape[1]):
            s = dev[:, col] @ dev[:, col]
            for j in range(1, L + 1):
                s += 2 * (1 - j / (L + 1)) * (dev[j:, col] @ dev[:-j, col])
            expected = np.sqrt(s / (T * (T - 1)))
            got = sp.fama_macbeth("y ~ x1 + x2", df, time="t", lags=L).std_errors.iloc[
                col
            ]
            np.testing.assert_allclose(got, expected, rtol=RTOL)

    def test_row_order_and_duplicate_index_labels_are_irrelevant(self):
        df = _panel()
        base = sp.fama_macbeth("y ~ x1 + x2", df, time="t")
        shuffled = df.sample(frac=1.0, random_state=1)
        relabelled = df.set_index(df.index % 5)
        for other in (shuffled, relabelled):
            res = sp.fama_macbeth("y ~ x1 + x2", other, time="t")
            np.testing.assert_allclose(
                res.params.to_numpy(), base.params.to_numpy(), rtol=RTOL
            )
            np.testing.assert_allclose(
                res.std_errors.to_numpy(), base.std_errors.to_numpy(), rtol=RTOL
            )

    def test_missing_period_rows_equal_prefiltering(self):
        df = _panel()
        t_missing = df["t"].astype(float).where(df.t != 2)
        a = sp.fama_macbeth("y ~ x1 + x2", df.assign(t=t_missing), time="t")
        b = sp.fama_macbeth("y ~ x1 + x2", df[df.t != 2], time="t")
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=RTOL)
        assert a.model_info["n_periods"] == 6

    def test_unidentified_period_is_skipped_with_a_warning(self):
        df = _panel()
        thin = pd.concat([df[df.t != 3], df[(df.t == 3) & (df.id < 2)]])
        with pytest.warns(UserWarning, match="left out"):
            res = sp.fama_macbeth("y ~ x1 + x2", thin, time="t")
        assert res.model_info["skipped_periods"] == [3]
        expected = sp.fama_macbeth("y ~ x1 + x2", df[df.t != 3], time="t")
        np.testing.assert_allclose(
            res.params.to_numpy(), expected.params.to_numpy(), rtol=RTOL
        )

    def test_refusals(self):
        df = _panel()
        with pytest.raises(DataInsufficient, match="at least two"):
            sp.fama_macbeth("y ~ x1 + x2", df[df.t == 0], time="t")
        with pytest.raises(DataInsufficient, match="lags"):
            sp.fama_macbeth("y ~ x1 + x2", df, time="t", lags=7)
        with pytest.raises(MethodIncompatibility, match="lags"):
            sp.fama_macbeth("y ~ x1 + x2", df, time="t", lags=-1)
        with pytest.raises(MethodIncompatibility, match="not a column"):
            sp.fama_macbeth("y ~ x1 + x2", df, time="year")


# ----------------------------------------------------------------------- xtsum


class TestXtsum:
    def test_overall_between_within_standard_deviations(self):
        """Stata's ``xtsum``: between is the SD of the unit means, within the
        SD of ``x - xbar_i + xbar``."""
        df = _panel().drop(index=[0, 1, 9])
        out = sp.xtsum(df, ["y"], id="id")
        means = df.groupby("id").y.mean()
        within = df.y - df.groupby("id").y.transform("mean") + df.y.mean()
        np.testing.assert_allclose(
            out.loc[("y", "overall"), "sd"], df.y.std(), rtol=RTOL
        )
        np.testing.assert_allclose(
            out.loc[("y", "between"), "sd"], means.std(), rtol=RTOL
        )
        np.testing.assert_allclose(
            out.loc[("y", "within"), "sd"], within.std(), rtol=RTOL
        )
        np.testing.assert_allclose(
            out.loc[("y", "within"), "min"], within.min(), rtol=RTOL
        )
        assert out.loc[("y", "overall"), "obs"] == len(df)
        assert out.loc[("y", "between"), "obs"] == df.id.nunique()

    def test_time_invariant_variable_has_no_within_variation(self):
        out = sp.xtsum(_panel(), ["z"], id="id")
        # rounding-level within SD against an overall SD of order one
        assert out.loc[("z", "within"), "sd"] < 1e-10 * out.loc[("z", "overall"), "sd"]


# -------------------------------------------------------------------- xtserial


class TestXtserial:
    @staticmethod
    def _by_hand(df):
        """Wooldridge's test on a panel without gaps, with pandas: FD
        regression, residual on its lag, cluster variance on the unit with
        G/(G-1) (one regressor, no constant)."""
        d = df.sort_values(["id", "t"])
        diff = d.groupby("id")[["y", "x1", "x2"]].diff().dropna()
        X = diff[["x1", "x2"]].to_numpy()
        beta = np.linalg.lstsq(X, diff["y"].to_numpy(), rcond=None)[0]
        e = pd.Series(diff["y"].to_numpy() - X @ beta, index=diff.index)
        unit = d.loc[diff.index, "id"]
        lag = e.groupby(unit).shift(1)
        ok = lag.notna()
        e, lag, unit = e[ok].to_numpy(), lag[ok].to_numpy(), unit[ok].to_numpy()
        rho = (lag @ e) / (lag @ lag)
        scores = pd.Series(lag * (e - rho * lag)).groupby(unit).sum().to_numpy()
        G = len(scores)
        var = (scores @ scores) / (lag @ lag) ** 2 * G / (G - 1)
        return (rho + 0.5) ** 2 / var, G, rho

    def test_statistic_by_hand(self):
        df = _panel()
        stat, G, rho = self._by_hand(df)
        out = sp.xtserial(df, "y", ["x1", "x2"], id="id", time="t")
        np.testing.assert_allclose(out["statistic"], stat, rtol=RTOL)
        np.testing.assert_allclose(out["rho"], rho, rtol=RTOL)
        assert out["df"] == (1, G - 1)
        np.testing.assert_allclose(out["pvalue"], stats.f.sf(stat, 1, G - 1), rtol=RTOL)

    def test_row_order_is_irrelevant(self):
        df = _panel().drop(index=[5, 6, 100])
        a = sp.xtserial(df, "y", ["x1", "x2"], id="id", time="t")
        b = sp.xtserial(
            df.sample(frac=1.0, random_state=2), "y", ["x1", "x2"], id="id", time="t"
        )
        np.testing.assert_allclose(a["statistic"], b["statistic"], rtol=RTOL)

    def test_too_short_and_fractional_time_are_refused(self):
        df = _panel()
        with pytest.raises(DataInsufficient):
            sp.xtserial(df[df.t < 2], "y", ["x1"], id="id", time="t")
        with pytest.raises(MethodIncompatibility, match="integers"):
            sp.xtserial(df.assign(t=df.t + 0.5), "y", ["x1"], id="id", time="t")

    def test_time_invariant_regressor_is_omitted(self):
        df = _panel()
        try:
            out = sp.xtserial(df, "y", ["x1", "x2", "z"], id="id", time="t")
        except StatsPAIError:
            return
        base = sp.xtserial(df, "y", ["x1", "x2"], id="id", time="t")
        np.testing.assert_allclose(out["statistic"], base["statistic"], rtol=1e-8)


# -------------------------------------------------------------------- xtoverid


class TestXtoverid:
    def test_rows_with_a_missing_cluster_are_dropped(self):
        df = _panel()
        g = df["g"].astype(float).where(df.g > 0)
        a = sp.xtoverid(df.assign(g=g), "y", ["x1", "x2"], id="id", cluster="g")
        b = sp.xtoverid(df[df.g > 0], "y", ["x1", "x2"], id="id", cluster="g")
        np.testing.assert_allclose(a["statistic"], b["statistic"], rtol=RTOL)
        assert a["n_clusters"] == 6

    def test_time_invariant_regressor_is_kept_but_not_tested(self):
        out = sp.xtoverid(_panel(), "y", ["x1", "x2", "z"], id="id")
        assert out["tested"] == ["x1", "x2"] and out["df"] == 2

    def test_cluster_equal_to_id_is_the_default(self):
        df = _panel()
        a = sp.xtoverid(df, "y", ["x1", "x2"], id="id")
        b = sp.xtoverid(df.assign(id2=df.id), "y", ["x1", "x2"], id="id", cluster="id2")
        np.testing.assert_allclose(a["statistic"], b["statistic"], rtol=RTOL)

    def test_no_within_variation_is_refused(self):
        with pytest.raises(MethodIncompatibility, match="varies within"):
            sp.xtoverid(_panel(), "y", ["z"], id="id")


# ------------------------------------------------------------------- xthtaylor


class TestHausmanTaylor:
    KW = dict(id="id", endog=["x1", "z2"])
    F = "y ~ x1 + x2 + z + z2"

    def test_row_order_is_irrelevant(self):
        df = _panel().drop(index=[0, 1, 30])
        a = sp.xthtaylor(self.F, df, **self.KW)
        b = sp.xthtaylor(self.F, df.sample(frac=1.0, random_state=4), **self.KW)
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-8)
        np.testing.assert_allclose(
            a.std_errors.to_numpy(), b.std_errors.to_numpy(), rtol=1e-8
        )

    def test_robust_is_clustering_on_the_unit(self):
        df = _panel()
        a = sp.xthtaylor(self.F, df, vce="robust", **self.KW)
        b = sp.xthtaylor(self.F, df.assign(unit=df.id), cluster="unit", **self.KW)
        np.testing.assert_allclose(
            a.std_errors.to_numpy(), b.std_errors.to_numpy(), rtol=RTOL
        )
        assert a.model_info["n_clusters"] == df.id.nunique()

    def test_within_coefficients_bound_the_time_varying_ones(self):
        """The time-varying endogenous regressor is instrumented by its own
        within deviation, and with a just-identified time-invariant block the
        Hausman-Taylor slopes of the time-varying regressors are the within
        estimates."""
        df = _panel()
        ht = sp.xthtaylor("y ~ x1 + x2 + z2", df, id="id", endog=["x1", "z2"])
        fe = _quiet(sp.panel, df, "y ~ x1 + x2", entity="id", time="t", method="fe")
        np.testing.assert_allclose(
            ht.params[["x2", "x1"]].to_numpy(),
            fe.params[["x2", "x1"]].to_numpy(),
            rtol=1e-8,
        )

    @pytest.mark.parametrize(
        "kw,match",
        [
            (dict(id="id", endog=["x1", "z2"], method="am"), "not available"),
            (dict(id="id", endog=["x1", "z2"], vce="hc1"), "vce="),
            (dict(id="id", endog=["x1", "z2"], method="amacurdy"), "time="),
            (dict(id="id", endog=[]), "empty"),
            (dict(id="id", endog=["q"]), "not regressors"),
            (dict(id="id", endog=["x1", "z", "z2"]), "under-identified"),
        ],
    )
    def test_refusals(self, kw, match):
        with pytest.raises(MethodIncompatibility, match=match):
            sp.xthtaylor(self.F, _panel(), **kw)

    def test_missing_unit_id_is_refused(self):
        df = _panel()
        ids = df["id"].astype(float).where(df.id > 1)
        with pytest.raises(MethodIncompatibility, match="identifier is missing"):
            sp.xthtaylor(self.F, df.assign(id=ids), **self.KW)

    def test_missing_cluster_value_is_refused_or_dropped(self):
        df = _panel()
        g = df["g"].astype(float).where(df.g > 0)
        d = df.assign(g=g)
        try:
            res = sp.xthtaylor(self.F, d, cluster="g", **self.KW)
        except StatsPAIError:
            return
        pre = sp.xthtaylor(self.F, d[d.g.notna()], cluster="g", **self.KW)
        np.testing.assert_allclose(
            res.std_errors.to_numpy(), pre.std_errors.to_numpy(), rtol=1e-8
        )


# -------------------------------------------------------------- panel unit root


def _series(seed: int = 0, n_units: int = 12, n_periods: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_units), n_periods)
    df = pd.DataFrame({"id": ids, "t": np.tile(np.arange(n_periods), n_units)}).assign(
        e=rng.normal(size=n_units * n_periods)
    )
    df["rw"] = df.groupby("id").e.cumsum()
    df["st"] = df.e + 0.3 * df.groupby("id").e.shift(1).fillna(0.0)
    return df


class TestPanelUnitRoot:
    @pytest.mark.parametrize("var", ["rw", "st"])
    @pytest.mark.parametrize("trend", ["c", "ct"])
    @pytest.mark.parametrize("lags", [0, 2])
    def test_ips_tbar_and_fisher_p_against_statsmodels(self, var, trend, lags):
        """Per-unit ADF regressions with the OLS residual variance
        (``convention='stata'``) are statsmodels' ``adfuller`` with a fixed
        lag; IPS averages the t statistics, the Fisher P statistic is
        ``-2 sum log p_i`` on MacKinnon p-values."""
        from statsmodels.tsa.stattools import adfuller

        df = _series()
        fits = [
            adfuller(g[var].to_numpy(), maxlag=lags, regression=trend, autolag=None)
            for _, g in df.groupby("id")
        ]
        ips = sp.panel_unitroot(df, var, id="id", time="t", trend=trend, lags=lags)
        np.testing.assert_allclose(
            ips.details["tbar"], np.mean([f[0] for f in fits]), rtol=1e-8
        )
        W = np.sqrt(12) * (ips.details["tbar"] - ips.details["E_tbar"])
        W /= np.sqrt(ips.details["Var_tbar"])
        np.testing.assert_allclose(ips.statistic, W, rtol=RTOL)
        np.testing.assert_allclose(ips.p_value, stats.norm.cdf(W), rtol=RTOL)
        fisher = sp.panel_unitroot(
            df, var, id="id", time="t", test="fisher", trend=trend, lags=lags
        )
        P = -2 * np.sum(np.log([f[1] for f in fits]))
        # MacKinnon's response surface evaluated twice; 1e-6 covers a p-value
        # of order 1e-12 passing through log
        np.testing.assert_allclose(fisher.details["P"], P, rtol=1e-6)
        np.testing.assert_allclose(
            fisher.details["p_P"], stats.chi2.sf(P, 24), rtol=1e-5
        )

    def test_row_order_is_irrelevant(self):
        df = _series()
        a = sp.panel_unitroot(df, "rw", id="id", time="t", lags=1)
        b = sp.panel_unitroot(
            df.sample(frac=1.0, random_state=0), "rw", id="id", time="t", lags=1
        )
        np.testing.assert_allclose(a.statistic, b.statistic, rtol=RTOL)

    def test_hadri_robust_statistic_by_hand(self):
        df = _series()
        T = 30
        lm = []
        for _, g in df.groupby("id"):
            r = g.st.to_numpy() - g.st.mean()
            lm.append(np.sum(np.cumsum(r) ** 2) / T**2 / (r @ r / (T - 1)))
        z = np.sqrt(12) * (np.mean(lm) - 1 / 6) / np.sqrt(1 / 45)
        res = sp.panel_unitroot(df, "st", id="id", time="t", test="hadri")
        np.testing.assert_allclose(res.statistic, z, rtol=RTOL)
        np.testing.assert_allclose(res.p_value, stats.norm.sf(z), rtol=RTOL)

    @pytest.mark.parametrize(
        "kw,exc",
        [
            ({"test": "breitung"}, ValueError),
            ({"trend": "zzz"}, MethodIncompatibility),
            ({"convention": "zzz"}, MethodIncompatibility),
            ({"test": "ips", "trend": "n"}, MethodIncompatibility),
        ],
    )
    def test_invalid_options_are_refused(self, kw, exc):
        with pytest.raises(exc):
            sp.panel_unitroot(_series(), "rw", id="id", time="t", **kw)

    def test_hadri_needs_a_balanced_panel(self):
        df = _series().drop(index=[0, 1])
        with pytest.raises(MethodIncompatibility, match="balanced"):
            sp.panel_unitroot(df, "st", id="id", time="t", test="hadri")

    def test_too_short_panels_are_refused(self):
        df = _series()
        with pytest.raises(DataInsufficient):
            sp.panel_unitroot(df[df.t < 4], "rw", id="id", time="t")

    @pytest.mark.parametrize("how", ["missing_period", "missing_value"])
    def test_interior_gap_is_refused(self, how):
        df = _series()
        if how == "missing_period":
            # a period the other units have (one absent from every unit is
            # not detectable: see TestPanelUnitRootSample)
            d = df[~((df.t == 12) & (df.id < 3))]
        else:
            d = df.assign(rw=df.rw.where(df.t != 12))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StatsPAIError):
                sp.panel_unitroot(d, "rw", id="id", time="t", lags=1)

    def test_negative_lags_are_refused_with_a_typed_error(self):
        with pytest.raises(StatsPAIError):
            sp.panel_unitroot(_series(), "rw", id="id", time="t", lags=-1)

    def test_default_lags_work_on_a_short_balanced_panel(self):
        df = _series(n_periods=7)
        res = sp.panel_unitroot(df, "rw", id="id", time="t")
        assert res.n_units == 12


# ---------------------------------------------------------------- panel logit


class TestPanelLogit:
    X = ["x1", "x2"]

    def test_missing_cluster_rows_are_dropped(self, method="re"):
        df = _panel(n_units=21, n_periods=5)
        g = df["g"].astype(float).where(df.index > 20)
        a = _quiet(
            sp.panel_logit, df.assign(g=g), "b", self.X, id="id", time="t",
            method=method, cluster="g", n_quadrature=6,
        )  # fmt: skip
        b = _quiet(
            sp.panel_logit, df[df.index > 20], "b", self.X, id="id", time="t",
            method=method, cluster="g", n_quadrature=6,
        )  # fmt: skip
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-6)
        np.testing.assert_allclose(
            a.std_errors.to_numpy(), b.std_errors.to_numpy(), rtol=1e-6
        )

    def test_fe_ignores_units_without_outcome_variation(self):
        """Conditional logit: a unit whose outcome never changes contributes
        nothing to the conditional likelihood."""
        df = _panel()
        stay = df.groupby("id").b.transform("nunique") == 1
        assert stay.any()
        a = _quiet(sp.panel_logit, df, "b", self.X, id="id", time="t", method="fe")
        b = _quiet(
            sp.panel_logit, df[~stay], "b", self.X, id="id", time="t", method="fe"
        )
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-6)
        np.testing.assert_allclose(
            a.std_errors.to_numpy(), b.std_errors.to_numpy(), rtol=1e-6
        )

    def test_fe_refuses_robust_and_cluster(self):
        df = _panel()
        for kw in ({"robust": "robust"}, {"cluster": "g"}):
            with pytest.raises(MethodIncompatibility, match="observed-information"):
                sp.panel_logit(df, "b", self.X, id="id", time="t", method="fe", **kw)

    def test_unknown_method_and_robust_are_refused(self):
        df = _panel()
        with pytest.raises(ValueError, match="method must be"):
            sp.panel_logit(df, "b", self.X, id="id", time="t", method="pooled")
        with pytest.raises(MethodIncompatibility, match="robust"):
            sp.panel_logit(
                df, "b", self.X, id="id", time="t", method="re", robust="zzz"
            )
        with pytest.raises(ValueError, match="FE probit"):
            sp.panel_probit(df, "b", self.X, id="id", time="t", method="fe")

    @pytest.mark.parametrize("coding", ["one_two", "continuous"])
    def test_non_binary_outcome_is_refused(self, coding):
        df = _panel()
        d = df.assign(b=df.b + 1) if coding == "one_two" else df.assign(b=df.y)
        with pytest.raises(StatsPAIError):
            _quiet(sp.panel_logit, d, "b", self.X, id="id", time="t", method="fe")

    def test_fe_time_invariant_regressor_is_omitted_or_refused(self):
        df = _panel(n_units=24, n_periods=5)
        try:
            res = _quiet(
                sp.panel_logit, df, "b", ["x1", "z"], id="id", time="t", method="fe"
            )
        except StatsPAIError:
            return
        assert "z" not in res.params.index or np.isnan(res.params["z"])


# -------------------------------------------------------------- interactive FE


class TestInteractiveFE:
    def test_row_order_is_irrelevant(self):
        df = _panel()
        a = _quiet(sp.interactive_fe, df, "y", ["x1", "x2"], id="id", time="t")
        b = _quiet(
            sp.interactive_fe,
            df.sample(frac=1.0, random_state=0),
            "y",
            ["x1", "x2"],
            id="id",
            time="t",
        )
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-7)

    def test_unbalanced_units_are_dropped_with_a_warning(self):
        df = _panel()
        with pytest.warns(RuntimeWarning, match="balanced"):
            a = sp.interactive_fe(
                df.drop(index=[3, 4]), "y", ["x1", "x2"], id="id", time="t"
            )
        b = _quiet(
            sp.interactive_fe, df[df.id != 0], "y", ["x1", "x2"], id="id", time="t"
        )
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-7)

    def test_unknown_method_and_dof_are_refused(self):
        df = _panel()
        with pytest.raises(MethodIncompatibility, match="method"):
            sp.interactive_fe(df, "y", ["x1"], id="id", time="t", method="zzz")
        with pytest.raises(MethodIncompatibility, match="dof"):
            sp.interactive_fe(df, "y", ["x1"], id="id", time="t", dof="zzz")

    def test_negative_number_of_factors_is_refused(self):
        with pytest.raises((StatsPAIError, ValueError)):
            _quiet(
                sp.interactive_fe, _panel(), "y", ["x1", "x2"], id="id", time="t",
                n_factors=-1,
            )  # fmt: skip


class TestPanelUnitRootSample:
    def test_gap_check_accepts_a_coarser_time_step(self):
        """Periods every four years are consecutive periods of that panel."""
        df = _series()
        a = sp.panel_unitroot(df, "rw", id="id", time="t", lags=1)
        b = sp.panel_unitroot(
            df.assign(t=1960 + 4 * df.t), "rw", id="id", time="t", lags=1
        )
        np.testing.assert_allclose(b.statistic, a.statistic, rtol=RTOL)

    def test_months_coded_yyyymm_across_a_year_boundary_are_consecutive(self):
        """200012 -> 200101 is one month: the codes jump by 89, the periods
        do not. Same statistic as the panel with a running month counter."""
        df = _series(n_periods=36)
        coded = df.assign(t=(2000 + df.t // 12) * 100 + df.t % 12 + 1)
        assert {200012, 200101} <= set(coded.t)
        a = sp.panel_unitroot(df, "rw", id="id", time="t", lags=1)
        b = sp.panel_unitroot(coded, "rw", id="id", time="t", lags=1)
        np.testing.assert_allclose(b.statistic, a.statistic, rtol=RTOL)
        np.testing.assert_allclose(b.p_value, a.p_value, rtol=RTOL)

    def test_a_skipped_month_that_other_units_have_is_refused(self):
        df = _series(n_periods=36)
        coded = df.assign(t=(2000 + df.t // 12) * 100 + df.t % 12 + 1)
        d = coded[~((coded.id == 4) & (coded.t == 200107))]
        with pytest.raises(MethodIncompatibility, match="1 gap"):
            sp.panel_unitroot(d, "rw", id="id", time="t", lags=1)

    def test_a_period_absent_from_every_unit_is_not_detected(self):
        """The documented limit of the rule: a period no unit has cannot be
        told from a coding jump, so the remaining periods are taken as
        consecutive. The result is the one for the relabelled panel."""
        df = _series()
        d = df[df.t != 12]
        relabelled = d.assign(t=d.groupby("id").cumcount())
        a = sp.panel_unitroot(relabelled, "rw", id="id", time="t", lags=1)
        b = sp.panel_unitroot(d, "rw", id="id", time="t", lags=1)
        np.testing.assert_allclose(b.statistic, a.statistic, rtol=RTOL)

    def test_late_start_is_not_a_gap(self):
        df = _series()
        d = df[~((df.id < 3) & (df.t < 6))]
        res = _quiet(sp.panel_unitroot, d, "rw", id="id", time="t", lags=1)
        assert res.n_units == 12

    def test_default_lag_rule_is_unchanged_where_it_was_feasible(self):
        """T = 30: floor(4 (T/100)^(1/4)) = 2 lags, as statsmodels with
        maxlag=2."""
        from statsmodels.tsa.stattools import adfuller

        df = _series()
        res = sp.panel_unitroot(df, "rw", id="id", time="t")
        tbar = np.mean(
            [
                adfuller(g.rw.to_numpy(), maxlag=2, regression="c", autolag=None)[0]
                for _, g in df.groupby("id")
            ]
        )
        np.testing.assert_allclose(res.details["tbar"], tbar, rtol=1e-8)

    def test_statistic_over_a_minority_of_units_is_refused(self):
        """lags=3 leaves the 6-period units without a feasible regression:
        2 usable units of 12."""
        df = _series()
        d = df[(df.id < 2) | (df.t < 6)]
        with pytest.raises(DataInsufficient, match="only 2 of 12"):
            _quiet(sp.panel_unitroot, d, "rw", id="id", time="t", lags=3)


class TestPanelLogitFEOmission:
    def test_omitted_regressor_leaves_the_other_coefficients_alone(self):
        df = _panel(n_units=24, n_periods=5)
        kw = dict(id="id", time="t", method="fe")
        with pytest.warns(UserWarning, match="no within-unit variation"):
            res = sp.panel_logit(df, "b", ["x1", "z"], **kw)
        base = _quiet(sp.panel_logit, df, "b", ["x1"], **kw)
        np.testing.assert_allclose(res.params["x1"], base.params["x1"], rtol=RTOL)
        assert res.model_info["omitted"][0]["variable"] == "z"
